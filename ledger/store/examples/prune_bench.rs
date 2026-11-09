// Copyright (c) 2019-2026 Provable Inc.
// This file is part of the snarkVM library.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:

// http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Times the startup certificate-index pass of `prune-authority-data` on a synthetic full-size
//! index, in several variants. Run with `--help` for the commands and the Linux setup.

use aleo_std_storage::StorageMode;
use anyhow::{Context, Result, anyhow, bail, ensure};
use snarkvm_console::{
    network::{MainnetV0, Network},
    prelude::FromBytes,
    types::Field,
};
use snarkvm_ledger_store::{
    AUTHORITY_RETENTION_BLOCKS,
    BlockStorage,
    helpers::rocksdb::{BlockDB, BlockMap, Database, MapID, PREFIX_LEN, RocksDB},
    oldest_retained_authority_height,
    prune_expired_authorities,
};
use snarkvm_utilities::bytes::unchecked_deserialize;
use std::{
    io::{BufRead, BufReader},
    path::{Path, PathBuf},
    sync::atomic::{AtomicUsize, Ordering},
    time::Instant,
};

type N = MainnetV0;

const HELP: &str = "\
prune_bench: times the startup certificate-index pass of prune-authority-data.

Commands (all take the database directory first):
  build-index <db_dir> <heights> <certs_per_height> <seed>
        Writes <certs_per_height> certificate index rows for every height in 1..<heights>, in height
        order, with random keys. Sets the authority-pruning watermark to the retained window, so the
        authority walk is a no-op, and waits for compaction to settle. Resumable: rerun the same
        command after an interruption.
  settle <db_dir>              Flushes and waits for background compaction to finish.
  status <db_dir>              Prints the watermark and level layout.
  scan <db_dir> [--raw]        Iterates the certificate map and counts the rows a pass would remove,
                               decoding keys and values as the store does. --raw decodes only the
                               height from the raw value bytes.
  prune <db_dir>               The production pass (`prune_expired_authorities`). Its log lines
                               give the scan+delete time and the compaction time separately.
  prune-parallel <db_dir> <threads> [--no-compact]
                               One forward iterator and direct point deletes per key-range thread,
                               then one compaction.
  rebuild <db_dir> [slices] [--threads N] [--no-compact]
                               Rewrites the map in key-range slices (default 256), each one range
                               delete plus puts of the kept rows, then one compaction.
  compact-certs <db_dir>       Compacts the certificate map's key range.
  restart <db_dir>             The production pass again, when there is nothing left to remove.

Flags for every command:
  --readahead <bytes>          Iterator readahead (RocksDB default 0: automatic, 8 KB to 256 KB per
                               file). Not applied to `prune`/`restart`, which use the store's iterator.
  --subcompactions <n>         Opens the store with max_subcompactions = n (via the env var
                               SNARKVM_ROCKSDB_MAX_SUBCOMPACTIONS), so a manual compaction can use
                               more than one thread. Default: the store's setting (1).

Every variant prints wall time per phase, rows scanned and removed, disk bytes read and written by
this process, write stalls found in the RocksDB LOG, and RocksDB properties before and after.
Logging: RUST_LOG=info is the default.

Ubuntu 22.04 setup:
  sudo apt-get install -y build-essential clang libclang-dev pkg-config libssl-dev
  curl https://sh.rustup.rs -sSf | sh -s -- -y && . ~/.cargo/env
Build:
  cargo build --release -p snarkvm-ledger-store --features rocks,prune-authority-data --example prune_bench
  BIN=target/release/examples/prune_bench
Dataset (about 570M rows, 21.3M heights x 27):
  $BIN build-index /data/certs 21300000 27 1
  snapshot or copy /data/certs now, and restore it before each variant
Variants, each on a fresh copy of the dataset:
  $BIN scan /data/certs; $BIN scan /data/certs --raw; $BIN scan /data/certs --readahead 2097152
  $BIN prune /data/certs;                                $BIN restart /data/certs
  $BIN prune-parallel /data/certs 1 --readahead 2097152; $BIN restart /data/certs
  $BIN prune-parallel /data/certs 8;                     $BIN restart /data/certs
  $BIN rebuild /data/certs --readahead 2097152;          $BIN restart /data/certs
  $BIN rebuild /data/certs --threads 8 --subcompactions 8; $BIN restart /data/certs
Restart cost before compaction: run a variant with --no-compact, then restart, then compact-certs
(optionally --subcompactions 8), then restart again.
";

/// A copy of the block store's private `is_pruned_height`.
fn is_pruned_height(below: u32, height: u32) -> bool {
    (1..below).contains(&height)
}

fn cert_prefix() -> [u8; PREFIX_LEN] {
    let mut prefix = [0u8; PREFIX_LEN];
    prefix[..2].copy_from_slice(&N::ID.to_le_bytes());
    prefix[2..].copy_from_slice(&u16::from(MapID::Block(BlockMap::Certificate)).to_le_bytes());
    prefix
}

fn cert_prefix_end() -> [u8; PREFIX_LEN] {
    (u32::from_be_bytes(cert_prefix()) + 1).to_be_bytes()
}

/// The key bytes after the map prefix at which slice `i` of `k` starts; `None` for `i == k`, the map's
/// end. Certificate IDs are uniform, so the first two key bytes split the map evenly.
fn slice_start(i: usize, k: usize) -> Option<[u8; 2]> {
    (i < k).then(|| u16::try_from(i * 65536 / k).expect("slice index below count").to_be_bytes())
}

/// The raw key bounds of slice `i` of `k`.
fn slice_bounds(i: usize, k: usize) -> (Vec<u8>, Vec<u8>) {
    let bound = |j: usize| match slice_start(j, k) {
        Some(start) => [&cert_prefix()[..], &start].concat(),
        None => cert_prefix_end().to_vec(),
    };
    (bound(i), bound(i + 1))
}

fn range_iter<'a>(db: &'a rocksdb::DB, lower: &[u8], upper: &[u8], readahead: usize) -> rocksdb::DBRawIterator<'a> {
    let mut options = rocksdb::ReadOptions::default();
    options.set_total_order_seek(true);
    options.set_iterate_lower_bound(lower.to_vec());
    options.set_iterate_upper_bound(upper.to_vec());
    if readahead > 0 {
        options.set_readahead_size(readahead);
    }
    let mut iter = db.raw_iterator_opt(options);
    iter.seek(lower);
    iter
}

/// The height from a raw `(u32, u64)` certificate index value.
fn raw_height(value: &[u8]) -> Result<u32> {
    Ok(u32::from_le_bytes(value.get(..4).and_then(|v| v.try_into().ok()).context("short certificate index value")?))
}

#[derive(Clone, Copy, Default)]
struct DiskIo {
    read: u64,
    write: u64,
}

#[cfg(target_os = "linux")]
fn disk_io() -> DiskIo {
    let mut io = DiskIo::default();
    for line in std::fs::read_to_string("/proc/self/io").unwrap_or_default().lines() {
        if let Some(v) = line.strip_prefix("read_bytes:") {
            io.read = v.trim().parse().unwrap_or(0);
        } else if let Some(v) = line.strip_prefix("write_bytes:") {
            io.write = v.trim().parse().unwrap_or(0);
        }
    }
    io
}

#[cfg(target_os = "macos")]
fn disk_io() -> DiskIo {
    // SAFETY: `proc_pid_rusage` fills a zeroed struct of the size its flavor requires.
    unsafe {
        let mut info: libc::rusage_info_v2 = std::mem::zeroed();
        let ptr = &mut info as *mut libc::rusage_info_v2 as *mut libc::rusage_info_t;
        if libc::proc_pid_rusage(libc::getpid(), libc::RUSAGE_INFO_V2, ptr) == 0 {
            DiskIo { read: info.ri_diskio_bytesread, write: info.ri_diskio_byteswritten }
        } else {
            DiskIo::default()
        }
    }
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn disk_io() -> DiskIo {
    DiskIo::default()
}

fn gb(bytes: u64) -> f64 {
    bytes as f64 / 1e9
}

fn stall_count(log: &Path) -> usize {
    let Ok(file) = std::fs::File::open(log) else { return 0 };
    BufReader::new(file)
        .lines()
        .map_while(Result::ok)
        .filter(|l| l.contains("Stalling writes") || l.contains("Stopping writes"))
        .count()
}

fn print_props(db: &RocksDB, label: &str, full: bool) {
    println!("--- rocksdb {label}");
    for p in [
        "rocksdb.num-running-compactions",
        "rocksdb.estimate-pending-compaction-bytes",
        "rocksdb.num-files-at-level0",
        "rocksdb.is-write-stopped",
        "rocksdb.actual-delayed-write-rate",
        "rocksdb.total-sst-files-size",
        "rocksdb.estimate-num-keys",
    ] {
        println!("{p} = {}", db.property_int_value(p).ok().flatten().unwrap_or(0));
    }
    let mut props = vec!["rocksdb.cf-write-stall-stats", "rocksdb.levelstats"];
    if full {
        props.push("rocksdb.stats");
    }
    for p in props {
        println!("{p}:\n{}", db.property_value(p).ok().flatten().unwrap_or_default());
    }
}

struct Bench {
    storage: BlockDB<N>,
    db: RocksDB,
    log: PathBuf,
    readahead: usize,
}

impl Bench {
    fn open(db_dir: &str, readahead: usize, subcompactions: Option<u32>) -> Result<Self> {
        if let Some(n) = subcompactions {
            // SAFETY: no other thread is running yet.
            unsafe { std::env::set_var("SNARKVM_ROCKSDB_MAX_SUBCOMPACTIONS", n.to_string()) };
            println!("max_subcompactions = {n}");
        }
        let mode = StorageMode::Custom(PathBuf::from(db_dir), None);
        let t = Instant::now();
        let storage = BlockDB::<N>::open(mode.clone())?;
        let db = RocksDB::open(N::ID, mode.clone())?;
        println!("opened {db_dir} in {:.2} s", t.elapsed().as_secs_f64());
        Ok(Self { storage, db, log: aleo_std_storage::aleo_ledger_dir(N::ID, &mode).join("LOG"), readahead })
    }

    fn watermark(&self) -> Result<u32> {
        self.storage.authority_pruned_below()?.context("no watermark: run build-index first")
    }

    fn raw_db(&self) -> &rocksdb::DB {
        &self.db
    }

    /// Runs `f`, printing wall time, disk bytes, write stalls and RocksDB state around it.
    fn timed<T>(&self, label: &str, f: impl FnOnce() -> Result<T>) -> Result<T> {
        print_props(&self.db, &format!("before {label}"), false);
        let (io0, stalls0, t) = (disk_io(), stall_count(&self.log), Instant::now());
        let out = f()?;
        let (secs, io1, stalls1) = (t.elapsed().as_secs_f64(), disk_io(), stall_count(&self.log));
        println!(
            "=== {label}: {secs:.1} s, disk read {:.2} GB, disk write {:.2} GB, write stalls {}",
            gb(io1.read - io0.read),
            gb(io1.write - io0.write),
            stalls1 - stalls0
        );
        print_props(&self.db, &format!("after {label}"), false);
        Ok(out)
    }

    fn compact(&self) -> Result<()> {
        self.timed("compaction", || self.storage.compact_certificate_map())
    }

    /// Waits for background compaction to finish, as on a node left running.
    fn settle(&self) -> Result<()> {
        self.db.flush()?;
        let t = Instant::now();
        loop {
            let pending = self.db.property_int_value("rocksdb.compaction-pending")?.unwrap_or(0);
            let running = self.db.property_int_value("rocksdb.num-running-compactions")?.unwrap_or(0);
            if pending == 0 && running == 0 {
                break;
            }
            std::thread::sleep(std::time::Duration::from_secs(5));
        }
        println!("compaction settled after {:.0} s", t.elapsed().as_secs_f64());
        print_props(&self.db, "settled", false);
        Ok(())
    }
}

/// SplitMix64, so each height's rows are reproducible on their own.
struct SplitMix(u64);

impl SplitMix {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

/// Fills `key` (after the map prefix) with a random valid field element in little-endian bytes.
fn random_key(rng: &mut SplitMix, key: &mut [u8]) {
    for chunk in key.chunks_exact_mut(8) {
        chunk.copy_from_slice(&rng.next().to_le_bytes());
    }
    // The BLS12-377 scalar modulus starts with 0x12, so a top byte below 0x10 is always in range.
    key[31] &= 0x0f;
}

fn build_index(bench: &Bench, db_dir: &str, heights: u32, certs: u32, seed: u64) -> Result<()> {
    ensure!(heights > AUTHORITY_RETENTION_BLOCKS, "heights must exceed the retention window");
    // The bench writes raw keys; make sure they match what `DataMap` would write.
    let mut sample = [0u8; 32];
    random_key(&mut SplitMix(seed), &mut sample);
    let field = Field::<N>::from_bytes_le(&sample)?;
    ensure!(bincode::serialize(&field)? == sample, "raw key encoding differs from bincode's");

    let progress_path = Path::new(db_dir).join("prune_bench_build.progress");
    let header = format!("{heights} {certs} {seed}");
    let mut start = 1u32;
    if let Ok(progress) = std::fs::read_to_string(&progress_path) {
        let (h, next) = progress.trim().rsplit_once(' ').context("malformed progress file")?;
        ensure!(h == header, "progress file {} is for different arguments", progress_path.display());
        start = next.parse()?;
        println!("resuming at height {start}");
    }
    let save_progress = |next: u32| std::fs::write(&progress_path, format!("{header} {next}"));

    let db = bench.raw_db();
    let mut write_options = rocksdb::WriteOptions::default();
    write_options.disable_wal(true);
    let mut batch = rocksdb::WriteBatch::default();
    let mut key = cert_prefix().to_vec();
    key.resize(PREFIX_LEN + 32, 0);
    let mut value = [0u8; 12];
    let (t, mut rows) = (Instant::now(), 0u64);
    for height in start..heights {
        let mut rng = SplitMix(seed ^ (u64::from(height) << 20));
        value[..4].copy_from_slice(&height.to_le_bytes());
        for i in 0..certs {
            random_key(&mut rng, &mut key[PREFIX_LEN..]);
            // Each block commits certificates of about two rounds.
            value[4..].copy_from_slice(&(u64::from(height) * 2 + u64::from(i % 2)).to_le_bytes());
            batch.put(&key, value);
        }
        rows += u64::from(certs);
        if height.is_multiple_of(1_000) {
            db.write_opt(std::mem::take(&mut batch), &write_options)?;
        }
        if height.is_multiple_of(500_000) {
            db.flush()?;
            save_progress(height + 1)?;
        }
        if height.is_multiple_of(100_000) {
            let secs = t.elapsed().as_secs_f64();
            println!("height {height}: {rows} rows, {:.0} rows/s, {secs:.0} s", rows as f64 / secs);
        }
    }
    db.write_opt(batch, &write_options)?;
    db.flush()?;
    println!("wrote {rows} rows in {:.0} s", t.elapsed().as_secs_f64());

    let floor = oldest_retained_authority_height(heights - 1);
    bench.storage.start_atomic();
    bench.storage.set_authority_pruned_below(floor)?;
    bench.storage.finish_atomic()?;
    save_progress(heights)?;
    println!("watermark = {floor}; retained heights {floor}..{heights}");
    bench.settle()
}

fn scan(bench: &Bench, raw: bool) -> Result<()> {
    let below = bench.watermark()?;
    let (t, mut scanned, mut removable) = (Instant::now(), 0usize, 0usize);
    let report = |scanned: usize, removable: usize| {
        if scanned.is_multiple_of(50_000_000) {
            let secs = t.elapsed().as_secs_f64();
            println!("scanned {scanned}, removable {removable}, {:.0} rows/s, {secs:.0} s", scanned as f64 / secs);
        }
    };
    bench.timed(if raw { "scan --raw" } else { "scan" }, || {
        let mut iter = range_iter(bench.raw_db(), &cert_prefix(), &cert_prefix_end(), bench.readahead);
        while let Some((key, value)) = iter.item() {
            scanned += 1;
            let height = if raw {
                raw_height(value)?
            } else {
                // The same decoding as the store's `iter_confirmed`.
                let _id: Field<N> = unchecked_deserialize(&key[PREFIX_LEN..])?;
                let index: (u32, u64) = unchecked_deserialize(value)?;
                index.0
            };
            removable += usize::from(is_pruned_height(below, height));
            report(scanned, removable);
            iter.next();
        }
        iter.status()?;
        Ok(())
    })?;
    println!("scanned {scanned} rows, {removable} removable, {:.0} rows/s", scanned as f64 / t.elapsed().as_secs_f64());
    Ok(())
}

/// The production path. `prune_expired_authorities` logs the scan+delete and compaction times.
fn prune(bench: &Bench) -> Result<()> {
    let below = bench.watermark()?;
    let latest_height = below + AUTHORITY_RETENTION_BLOCKS - 1;
    bench.timed("prune (production path)", || prune_expired_authorities(&bench.storage, latest_height))
}

fn prune_parallel(bench: &Bench, threads: usize, compact: bool) -> Result<()> {
    let below = bench.watermark()?;
    let (scanned, removed) = (AtomicUsize::new(0), AtomicUsize::new(0));
    bench.timed(&format!("prune-parallel x{threads}"), || {
        std::thread::scope(|scope| {
            let workers: Vec<_> = (0..threads)
                .map(|i| {
                    let (lower, upper) = slice_bounds(i, threads);
                    let (scanned, removed) = (&scanned, &removed);
                    scope.spawn(move || -> Result<()> {
                        let db = bench.raw_db();
                        let mut iter = range_iter(db, &lower, &upper, bench.readahead);
                        let mut batch = rocksdb::WriteBatch::default();
                        let (mut n, mut r) = (0usize, 0usize);
                        while let Some((key, value)) = iter.item() {
                            n += 1;
                            if is_pruned_height(below, raw_height(value)?) {
                                batch.delete(key);
                                r += 1;
                                if batch.len() == 10_000 {
                                    db.write(std::mem::take(&mut batch))?;
                                }
                            }
                            iter.next();
                        }
                        iter.status()?;
                        db.write(batch)?;
                        println!("thread {i}: scanned {n}, removed {r}");
                        scanned.fetch_add(n, Ordering::Relaxed);
                        removed.fetch_add(r, Ordering::Relaxed);
                        Ok(())
                    })
                })
                .collect();
            for worker in workers {
                worker.join().map_err(|_| anyhow!("worker thread panicked"))??;
            }
            Ok(())
        })
    })?;
    println!("scanned {} rows, removed {}", scanned.load(Ordering::Relaxed), removed.load(Ordering::Relaxed));
    if compact { bench.compact() } else { Ok(()) }
}

fn rebuild(bench: &Bench, slices: usize, threads: usize, compact: bool) -> Result<()> {
    ensure!((1..=65536).contains(&slices), "slices must be in 1..=65536");
    let below = bench.watermark()?;
    let (kept, removed, next) = (AtomicUsize::new(0), AtomicUsize::new(0), AtomicUsize::new(0));
    bench.timed(&format!("rebuild {slices} slices x{threads} threads"), || {
        std::thread::scope(|scope| {
            let workers: Vec<_> = (0..threads)
                .map(|_| {
                    let (kept, removed, next) = (&kept, &removed, &next);
                    scope.spawn(move || -> Result<()> {
                        loop {
                            let i = next.fetch_add(1, Ordering::Relaxed);
                            if i >= slices {
                                return Ok(());
                            }
                            let t = Instant::now();
                            let start = slice_start(i, slices).expect("slice below count");
                            let end = slice_start(i + 1, slices);
                            let (k, r) = bench.storage.certificate_map().retain_range(
                                &start,
                                end.as_ref().map(|e| &e[..]),
                                bench.readahead,
                                |_, (height, _)| !is_pruned_height(below, *height),
                            )?;
                            if slices <= 64 || i.is_multiple_of(slices / 16) {
                                println!("slice {i}: kept {k}, removed {r}, {:.1} s", t.elapsed().as_secs_f64());
                            }
                            kept.fetch_add(k, Ordering::Relaxed);
                            removed.fetch_add(r, Ordering::Relaxed);
                        }
                    })
                })
                .collect();
            for worker in workers {
                worker.join().map_err(|_| anyhow!("worker thread panicked"))??;
            }
            Ok(())
        })
    })?;
    let (kept, removed) = (kept.load(Ordering::Relaxed), removed.load(Ordering::Relaxed));
    println!("scanned {} rows, kept {kept}, removed {removed}", kept + removed);
    if compact { bench.compact() } else { Ok(()) }
}

fn flag(args: &[String], name: &str) -> bool {
    args.iter().any(|a| a == name)
}

fn flag_value<T: std::str::FromStr>(args: &[String], name: &str) -> Result<Option<T>>
where
    T::Err: std::error::Error + Send + Sync + 'static,
{
    match args.iter().position(|a| a == name) {
        Some(i) => Ok(Some(args.get(i + 1).with_context(|| format!("{name} needs a value"))?.parse()?)),
        None => Ok(None),
    }
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()))
        .with_target(false)
        .init();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(command) = args.first() else { bail!("{HELP}") };
    let arg = |i: usize| args.get(i).with_context(|| format!("missing argument {i}; see --help"));
    if command == "--help" || command == "-h" || command == "help" {
        print!("{HELP}");
        return Ok(());
    }
    let db_dir = arg(1)?;
    let bench =
        Bench::open(db_dir, flag_value(&args, "--readahead")?.unwrap_or(0), flag_value(&args, "--subcompactions")?)?;
    if command == "build-index" {
        return build_index(&bench, db_dir, arg(2)?.parse()?, arg(3)?.parse()?, arg(4)?.parse()?);
    }
    print_props(&bench.db, "at open", true);
    match command.as_str() {
        "settle" => bench.settle()?,
        "status" => {
            println!("watermark = {:?}", bench.storage.authority_pruned_below()?);
        }
        "scan" => scan(&bench, flag(&args, "--raw"))?,
        "prune" | "restart" => prune(&bench)?,
        "prune-parallel" => prune_parallel(&bench, arg(2)?.parse()?, !flag(&args, "--no-compact"))?,
        "rebuild" => {
            let slices = match args.get(2).filter(|a| !a.starts_with("--")) {
                Some(slices) => slices.parse()?,
                None => 256,
            };
            rebuild(&bench, slices, flag_value(&args, "--threads")?.unwrap_or(1), !flag(&args, "--no-compact"))?
        }
        "compact-certs" => bench.compact()?,
        other => bail!("unknown command {other}; see --help"),
    }
    print_props(&bench.db, "at exit", true);
    Ok(())
}
