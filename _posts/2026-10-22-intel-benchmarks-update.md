---
#### Blog Post Template ####

#### Post Information ####
title: "Accelerating scikit-learn on Intel hardware: what the benchmarks found so far"
date: October 22, 2026

#### Post Category and Tags ####
# Format in titlecase without dashes (Ex. "Open Source" instead of "open-source")
categories:
  - Updates
tags:
  - Performance
  - Benchmarks
  - Open Source
  - Machine Learning

#### Author Info ####
# Can accomodate multiple authors
# Add SQUARE Author Image to /assets/images/author_images/ folder
postauthors:
  - name: Arthur Lacote
    website: https://github.com/cakedev0
    image: arthur_lacote.png
---
<div>
  {% include postauthor.html %}
</div>

**TL;DR:**

- We care about the performance and resource use of scikit-learn, and we're
  glad that Probabl and Intel collaborate to improve them. As a first step,
  we built a benchmark suite for scikit-learn on Intel hardware, and focused
  on a few important estimators: linear models, tree-based models and
  `KMeans`.
  The [code](https://github.com/probabl-ai/scikit-learn-benchmarks) and the
  [results](https://probabl-ai.github.io/scikit-learn-benchmarks/snapshots/v1.rc/)
  are public.
- The good news: these benchmarks are driving performance improvements in
  scikit-learn and in other projects of the ecosystem (OpenBLAS, oneDAL,
  PyTorch, joblib, loky). Once released, these improvements reach users
  without any change on their side, and not only on Intel hardware.
- If you run scikit-learn on a machine with many cores, the benchmark results
  are worth a closer look. The main tip: set `n_jobs` on the outer loop
  (`GridSearchCV`, `cross_validate`...) rather than counting on a single fit
  to use all the cores. A single fit rarely uses them well:
  `HistGradientBoosting`, for instance, uses all of them by default, and can
  then be up to 30x slower than with fewer threads (a fix giving 2x to 10x
  speed-ups on small and medium datasets is under review).
- If you need speed now, without waiting for these improvements to land in
  scikit-learn, have a look at
  [scikit-learn-intelex](https://uxlfoundation.github.io/scikit-learn-intelex/2026.2/),
  an extension from the [UXL Foundation](https://uxlfoundation.org/), initially
  developed by Intel, that swaps in faster implementations while keeping the
  scikit-learn API. Among what we benchmarked, it's the fastest option today
  for tree-based models and `KMeans`. For linear models, the fastest is the
  [MKL build of scikit-learn](#for-linear-models-pick-the-mkl-build).

In June, at VivaTech, Probabl and Intel
[announced a collaboration](https://blog.probabl.ai/intel-and-probabl-announce-collaboration-to-accelerate-scikit-learn-on-intel-hardware)
to make scikit-learn faster on Intel hardware: Xeon CPUs, Intel GPUs, from
high performance data center servers with hundreds of cores down to consumer
laptops. We prioritized data center environments and use cases, where
performance directly translates into cost. If a fit only uses 4 cores of a 172-core server, you pay
for the other 168 for nothing.

> *"Enterprises run scikit-learn at scale while performance and cost directly
> shape what they can build. Our work with Intel is about delivering
> measurable gains in both, openly and for everyone, so the whole community
> moves forward together."* —Yann Lechelle, Executive President of Probabl

The Probabl Labs team has been working on these performance improvements
since April 2026, and this post
is a first progress update. So far, the core of the outcome is a
representative benchmark suite, surfacing most important problems and their
root causes: in scikit-learn, in
[scikit-learn-intelex](https://uxlfoundation.github.io/scikit-learn-intelex/2026.2/),
in interactions between BLAS libraries
and OpenMP runtimes, in PyTorch. Below, we explain how the benchmarks work, what they show,
and what we're now fixing because of them.

## Where we are after five months

In short:

- A benchmark repository and its
  [dashboards](https://probabl-ai.github.io/scikit-learn-benchmarks/snapshots/v1.rc/),
  which also check the impact of scikit-learn PRs on Intel machines before
  they get merged.
- ~20 issues reported to other projects: 11 to scikit-learn-intelex and
  oneDAL, the others to PyTorch's XPU backend, OpenBLAS, joblib, loky,
  narwhals and conda-forge. Some came with a fix, like the three
  merged in oneDAL.
- In scikit-learn, merged speed-ups for the encoders and for trees on
  conda-forge, and open PRs for `HistGradientBoosting`, `KMeans` and
  pandas categorical columns in encoders.
- Around the benchmarks, work by others in the collaboration: official Intel
  GPU support through the Python array API in scikit-learn 1.9, threadpoolctl 3.7.0,
  and thread management in joblib and loky (more at the end of this post).

Most of the scikit-learn changes are still under review. So far, the value is
mostly in knowing where the problems are, and in fixing them where they live, which is
sometimes not scikit-learn. All of it happens in the open, in the upstream
projects, under their own open source licenses, so every user of these
projects benefits, whatever their hardware vendor. The fixes that land in
scikit-learn reach users when they upgrade: same API, faster on the hardware
they already have.

## Why a new benchmark suite?

Before optimizing anything, we had to run some benchmarks with three goals in
mind:

- Profile realistic cases, looking for inefficiencies where scikit-learn
  shows room for improvement
- Reliably measure the impact of optimizations in scikit-learn before merging
  them
- Show users the trade-offs: which workloads benefit from
  scikit-learn-intelex, a GPU, a different build or a bigger machine, and
  when a comparison isn't like-for-like

Existing benchmarks were addressing different performance questions or were
meant as functionality tests. For instance, the
[scikit-learn workload of the Phoronix Test Suite](https://openbenchmarking.org/test/pts/scikit-learn),
which many hardware reviews use, runs the benchmark scripts of scikit-learn
1.2.2. Some of them scale extremely poorly: `HistGradientBoosting` gets up to
~20x slower on a 64-core server than on a laptop. That says
more about the benchmark (and about thread management, more on that below)
than about the hardware.

Intel's [scikit-learn_bench](https://github.com/IntelPython/scikit-learn_bench)
was a much better starting point. I started from a fork of it, and parts of
our code still derive from it, but we needed things it wasn't designed for:
comparing builds of the same library, measuring how a fit scales with the
number of cores, and benchmarking a scikit-learn pull request before it gets
merged. So it grew into its own project.

## How the benchmarks work

The suite fits and predicts with a restricted set of important estimators:

- Linear models:
  [`LinearRegression`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html),
  [`Ridge`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)
  and
  [`LogisticRegression`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- Tree-based models:
  [`RandomForest*`](https://scikit-learn.org/stable/modules/ensemble.html#random-forests),
  [`ExtraTrees*`](https://scikit-learn.org/stable/modules/ensemble.html#extremely-randomized-trees)
  and
  [`HistGradientBoosting*`](https://scikit-learn.org/stable/modules/ensemble.html#histogram-based-gradient-boosting)
  (classifiers and regressors)
- Clustering:
  [`KMeans`](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)

They run on two kinds of data. Synthetic datasets cover a matrix of shapes
(many samples, many features, tiny, large...) while keeping the total runtime
reasonable. Around 15 publicly and freely available
[real datasets](https://github.com/probabl-ai/scikit-learn-benchmarks/blob/main/configs/_real_datasets.py)
(covtype, SUSY, Ames housing, KDD Cup 2009, Amazon employee access...) come
with realistic preprocessing, including categorical columns and missing
values, and with hyperparameters picked by a small random search instead of
library defaults.

Each case runs in several software environments, all locked with
[pixi](https://pixi.sh):

- Builds of scikit-learn: the PyPI wheel, and conda-forge with MKL or with
  OpenBLAS (native "pthreads" threading backend or combined with LLVM OpenMP).
- [scikit-learn-intelex](https://uxlfoundation.github.io/scikit-learn-intelex/latest/)
  (`sklearnex`), on CPU and Intel GPU. It's an extension, initially developed
  by Intel, that replaces some scikit-learn estimators with accelerated
  implementations behind the same API, and falls back to scikit-learn when it
  doesn't support some parameters.
- [Python array API](https://scikit-learn.org/stable/modules/array_api.html)
  backends: [PyTorch](https://pytorch.org/) (CPU,
  [Intel XPU](https://docs.pytorch.org/docs/stable/notes/get_start_xpu.html),
  [NVIDIA CUDA](https://docs.pytorch.org/docs/stable/notes/cuda.html),
  [Apple MPS](https://docs.pytorch.org/docs/stable/notes/mps.html)),
  [dpnp](https://github.com/IntelPython/dpnp) and [CuPy](https://cupy.dev/).

The two main machines are provided by Intel:

- A high-end two-socket server with Intel Xeon 6787P CPUs (172 cores in total, 344
  with hyper-threading, 500GB of RAM)
- A laptop with an Intel Core Ultra X7 358H CPU (16 cores) and its integrated
  Arc B390 GPU


The hard part is comparing fairly. By default, `sklearnex` random forests bin
the features, which scikit-learn doesn't implement yet. Solvers stop after
different numbers of iterations. Some backends silently fall back to
scikit-learn for unsupported parameters. Hiding or dropping these cases would bias the
results. So each run records its metrics (ROC AUC, R2, inertia) and setup,
and the dashboards flag every case where the setup or the metrics differ
from the baseline. Each run also records the hardware and the software stack
(versions, BLAS and OpenMP runtimes).

The results are split in several dashboards, each varying one thing and
keeping the rest fixed:

- [Implementations](https://probabl-ai.github.io/scikit-learn-benchmarks/snapshots/v1.rc/per_hardware.html):
  scikit-learn vs `sklearnex` vs Python array API backends, on the same machine
- [Builds](https://probabl-ai.github.io/scikit-learn-benchmarks/snapshots/v1.rc/builds_comparison.html):
  the same scikit-learn code with different BLAS and OpenMP runtimes
- [Hardware](https://probabl-ai.github.io/scikit-learn-benchmarks/snapshots/v1.rc/hardware_comparisons.html):
  the same software on different machines
- [Core-count scalability](https://probabl-ai.github.io/scikit-learn-benchmarks/snapshots/v1.rc/models_scalability.html):
  how the time of a single fit changes with the number of cores
- [Hyperparameter search scalability](https://probabl-ai.github.io/scikit-learn-benchmarks/snapshots/v1.rc/hptuning_scalability.html):
  how a `RandomizedSearchCV` scales with its outer `n_jobs`
- [`HistGradientBoosting` breakdown](https://probabl-ai.github.io/scikit-learn-benchmarks/snapshots/v1.rc/hgb_scaling.html):
  which phases of an HGB fit scale with threads and which don't

Every point can be hovered, and links to a row in a table with the exact
timings, metrics and warnings.

![Implementations dashboard: speed-up of each implementation vs the scikit-learn PyPI wheel, for linear model fits](/assets/images/posts_images/intel-benchmarks/linear_implem_dashboard.png)

*Implementations dashboard on the laptop: speed-up of each implementation over
the scikit-learn PyPI wheel, for linear model fits. Each point is one case
(estimator, dataset, parameters).*

I also started using the same machinery to benchmark scikit-learn pull
requests, through some automation. This is for instance what I used to check
that preserving F-ordered data in `LogisticRegression` helps
([example](https://github.com/probabl-ai/scikit-learn-benchmarks/pull/100)).


## What we found

### scikit-learn-intelex is the most consistently fast option on CPU

`sklearnex` replaces some scikit-learn estimators with implementations from
the oneDAL library (both are now part of the UXL Foundation), and keeps the
scikit-learn API. Among everything we
benchmarked, it's the option that speeds up the most cases on CPU. Tree-based
models and `KMeans` gain the most: `KMeans` fits are ~3x faster and never
slower than with scikit-learn. Linear models gain less.

It also makes much better use of big machines. With scikit-learn, random
forests and extra trees fit ~2x to 4x faster on the 172-core server than on the
laptop. With `sklearnex`, they fit ~10x to 15x faster.

Part of the tree speed-up comes from binning: `sklearnex` uses
`max_bins=256` by default, which roughly doubles or triples the fit speed-up.
When both libraries use the same exact-split algorithm, the gap is much
smaller. This is a strong argument for adding binning to scikit-learn's
trees, see the last section.

The caveat is that `sklearnex` only supports
[a subset of estimators and parameters](https://uxlfoundation.github.io/scikit-learn-intelex/latest/algorithms.html),
and doesn't always behave exactly like scikit-learn. Running it on real
datasets, with missing values, categorical features and multiclass targets,
surfaced a number of bugs. I reported 11 issues to
[scikit-learn-intelex](https://github.com/uxlfoundation/scikit-learn-intelex)
and oneDAL, and sent fixes for some
of the tree ones. The most impactful:

- `ExtraTreesRegressor` stopped splitting too early
  ([oneDAL#3648](https://github.com/uxlfoundation/oneDAL/issues/3648), fixed
  in [oneDAL#3649](https://github.com/uxlfoundation/oneDAL/pull/3649)).
- Best-first tree growth (`max_leaf_nodes`) expands leaves in a worse order
  than scikit-learn
  ([oneDAL#3771](https://github.com/uxlfoundation/oneDAL/issues/3771), fixed
  in [oneDAL#3772](https://github.com/uxlfoundation/oneDAL/pull/3772) and
  [oneDAL#3773](https://github.com/uxlfoundation/oneDAL/pull/3773)).
- Tree models crash when `X` has missing values at predict time but not at fit
  time
  ([sklearnex#3356](https://github.com/uxlfoundation/scikit-learn-intelex/issues/3356),
  fixed in
  [sklearnex#3357](https://github.com/uxlfoundation/scikit-learn-intelex/pull/3357)
  and
  [sklearnex#3358](https://github.com/uxlfoundation/scikit-learn-intelex/pull/3358)).
- On laptops, multinomial `LogisticRegression` on 4 threads or more stops too
  early and returns a worse model
  ([oneDAL#3820](https://github.com/uxlfoundation/oneDAL/issues/3820), with a
  fix proposed in [oneDAL#3912](https://github.com/uxlfoundation/oneDAL/pull/3912)). This
  needs to be fixed before any performance measurements.
- CPU `Ridge` is much slower than scikit-learn when
  `n_features > n_samples`, because it doesn't use the dual formulation like
  scikit-learn does
  ([sklearnex#3377](https://github.com/uxlfoundation/scikit-learn-intelex/issues/3377),
  with a fix proposed in [oneDAL#3899](https://github.com/uxlfoundation/oneDAL/pull/3899)).

The Intel team has been quick to respond on all of them and fixes are either
already merged or on the way for most of them.

### For linear models, pick the MKL build

This is the cheapest speed-up in this post. With conda-forge's MKL build of
scikit-learn, BLAS-bound linear model fits are commonly 1.3x to 2x faster
than with the PyPI wheel
([Builds dashboard](https://probabl-ai.github.io/scikit-learn-benchmarks/snapshots/v1.rc/builds_comparison.html#high_end_intel_xeon_server)). You don't need to change
any code, only how you install scikit-learn:

```bash
conda install -c conda-forge scikit-learn "libblas=*=*mkl"
# or, with pixi
pixi add scikit-learn "libblas=*=*mkl"
```

It's also possible to install an MKL build
[with pip, from Intel's index](https://github.com/intel/optimization-zone/tree/main/software/scikit-learn#blas-and-lapack).

The PyPI build had a much bigger problem with `LogisticRegression`, now fixed
upstream. On the
laptop, every conda-forge build fits it much faster than the PyPI one, up to
~10x on some datasets. With PyPI, it even gets slower as soon as it runs on
more than one logical CPU. The cause is the OpenBLAS version bundled with
scipy 1.18: it wakes all its threads for the tiny triangular solves inside
L-BFGS-B, and synchronizing them costs much more than the work itself. This
was fixed in a more recent OpenBLAS, which scipy now ships
([scipy#26193](https://github.com/scipy/scipy/pull/26193#issuecomment-5886021070)).
The fix is in scipy 2.0, and should be backported to 1.18.2.

A smaller one, but in scikit-learn itself: `ExtraTrees` fits were ~25% slower
on conda-forge than with PyPI. conda-forge compiles against the headers of an
old glibc (2.17), where `isnan` is a macro that calls a glibc function instead
of the compiler builtin. So every `isnan` in the hot loops of the trees became
a real function call
([analysis in #34869](https://github.com/scikit-learn/scikit-learn/issues/34869)).
[#34876](https://github.com/scikit-learn/scikit-learn/pull/34876) replaces it
with `x != x` and will be in the next release. Other conda-forge packages with
`isnan` in a hot loop are probably affected too, so I
[reported it to conda-forge](https://github.com/conda-forge/ctng-compiler-activation-feedstock/issues/108).

This shows why it is valuable to run benchmarks for the same code installed
from alternative packaging options. The same scikit-learn version can be 10x
slower or faster depending on how it was installed.

### Big servers are built for many fits at once, not for one

For a single fit, the 172-core server mostly helps random forests and extra
trees. For linear models, the laptop can be faster: performance cores in
modern laptops are fast, and a single fit doesn't use many cores well. `Ridge`,
`LogisticRegression` and `KMeans` with scikit-learn stop scaling after a few
cores, and some get slower with more cores on the server.

Hyperparameter searches are where the server pays off. With many candidates
fitted in parallel, it gets through fits ~10x to 15x faster than the laptop,
even for linear models. That's close to the ratio of physical cores (172 vs
16).

The two levels of parallelism (candidates in parallel, threads inside each
fit) don't always cooperate. joblib limits the number of OpenMP and BLAS
threads in its worker processes, but not the threads that random forests
start with joblib's threading backend. On the server, a search over
`RandomForestClassifier` gets up to ~1.8x slower beyond ~22 outer workers
because of this oversubscription. This won't be the case in the next joblib
release: [joblib#1825](https://github.com/joblib/joblib/pull/1825) now limits
these nested calls too.

So if you work on a machine with many cores, parallelize the outer loop (hyperparameter
search, cross-validation) instead of counting on a single fit to use all the
cores. On a 172-core server, that's the difference between using most of the
machine and using a few percent of it.

### `HistGradientBoosting`: the best number of threads depends on the data

This is the clearest example of a thread-management problem. On the Xeon
server, the best number of threads for a `HistGradientBoosting*` fit depends
on the shape of the data: 1 thread for the smallest workloads, 4 to 8 for
medium ones, 16 to 64 for the largest. Past that point, fits get slower: every workload is 2x to 30x slower
on 128 or 172 threads than on its best thread count. And by default, HGB uses
all the cores of the machine.

![`HistGradientBoosting` fit time by phase vs number of threads, on `year_prediction_msd` and `covtype` for a Xeon server](/assets/images/posts_images/intel-benchmarks/hgb_dashboard.png)
![Legend: bin fit, bin transform, other / unmeasured, apply split, find split, compute hist](/assets/images/posts_images/intel-benchmarks/hgb_dashboard_legend.png)

*`HistGradientBoosting` fit time on the Xeon server, split by phase, for two
real datasets of ~500k samples. One can observe that fit duration decreases
when going from 1 to 8 or 16 threads depending on the shape of the data, and
beyond that point thread scheduling overhead dominates to the point of
degrading performance, especially on many-core hosts.*

In short, a single HGB fit typically runs thousands of small OpenMP
parallel regions
whatever the size of the data, and with many threads, starting a region might cost
more than the work inside it. That cost also depends on the OpenMP runtime and
settings, which differ between builds. On the laptop, the conda-forge builds get worse slowdowns because
they use a more recent OpenMP runtime than the PyPI wheels. I wrote down what
I know about it in [#34764](https://github.com/scikit-learn/scikit-learn/issues/34764).

[#34935](https://github.com/scikit-learn/scikit-learn/pull/34935) implements
different strategies to better utilize OpenMP threads. In our benchmarks, it
gives 2x to 10x speed-ups for small and medium datasets on the server and on
laptops with conda-forge-like OpenMP settings. It's still under review, and the
full story will get its own blog post.

### GPUs and the Python array API: early results

Intel GPUs are officially supported through the array API since scikit-learn
1.9, with PyTorch's XPU device and with dpnp. Only `LogisticRegression` and `Ridge`
are benchmarked on GPU for now, and here is where we stand:

- `LogisticRegression` on GPU is fairly fast.
- `Ridge` uses the SVD solver under the array API instead of Cholesky, which
  makes it much slower than the Cholesky solver used by default on CPU.
  [#35060](https://github.com/scikit-learn/scikit-learn/issues/35060) proposes
  adding array API support to the Cholesky solver.

The benchmarks also found bugs in PyTorch's XPU backend, for instance a
[50x to 80x slowdown in `matmul`](https://github.com/intel/torch-xpu-ops/issues/4805)
once the reduction dimension exceeds 2^24 (fixed), and
[bad precision for float64 matrix-vector products](https://github.com/pytorch/pytorch/issues/182282)
(fixed). These kinds of fixes are upstreamed regularly, so we advise users to
upgrade to the most recent PyTorch often to benefit from them. For instance,
the latest release at the time of writing (2.14) doesn't include all the
fixes listed above.

Other issues found and fixed along the way:
[an OpenBLAS crash](https://github.com/OpenMathLib/OpenBLAS/issues/5958) with
`KMeans` on the 344-logical-CPU server,
[wrong physical core counts in loky](https://github.com/joblib/loky/issues/639)
under CPU affinity.

## What's next

For the next months, our priorities are:

1. Improve the scalability of `HistGradientBoosting`, and make the new tools
   developed for this available in `threadpoolctl`
2. Turn the findings into performance guidance in the scikit-learn
   documentation
3. Add binning to random forests and extra trees, which should speed them up
   considerably
4. Finish the encoder speed-ups, which turned out to be a
   bottleneck in some pipelines
5. Extend the benchmarks

The biggest pieces will get their own blog post. Here is where each one
stands.

**Documentation.** Many of the findings above (which build to install, how
to set threads, using the `category` dtype) belong in the
[performance](https://scikit-learn.org/stable/computing/computational_performance.html)
and [parallelism](https://scikit-learn.org/stable/computing/parallelism.html)
guides of the scikit-learn documentation, with a section for Intel users,
from laptops to data center servers.

**`HistGradientBoosting` scalability.** See
[#34935](https://github.com/scikit-learn/scikit-learn/pull/34935) above.

**`KMeans`.** Looking at why `sklearnex` is so much faster on `KMeans` led to
two PRs, still under review, that look like the most promising speed-ups so
far. On the Xeon server, the k-means++ initialization took 75% to 97% of the
fit time: for float32 data, it went through many small float64 BLAS calls,
and for low-dimensional data, single-threaded NumPy passes dominated.
[#35140](https://github.com/scikit-learn/scikit-learn/pull/35140) computes
each step of the initialization in a single OpenMP kernel. In our
benchmarks, on one socket of the server, the initialization gets ~8x faster
with OpenBLAS and ~20x faster with MKL (median over 34 cases), and ~10x
faster on the laptop. The Lloyd iterations that follow stopped scaling at
~16 threads, because of costs that grow with the number of threads: a lock
to merge the per-thread buffers, the GIL taken by every thread at each
iteration, and buffers allocated at each iteration.
[#35193](https://github.com/scikit-learn/scikit-learn/pull/35193) removes
them. In our benchmarks, the iterations get ~4x to ~6x faster (median) on one
socket of the server, and up to 29x on sparse data. On the laptop, with fewer
threads, it's on par with `main` in most cases.

If they are merged, they should also close a long-standing issue,
[#17334](https://github.com/scikit-learn/scikit-learn/issues/17334), opened in
2020: `KMeans` with OpenBLAS getting much slower with more threads. On a
recent Intel laptop, the reproducer of that issue takes ~65 s with the
scikit-learn 1.9.1 wheel and ~10 s with both PRs. It drops under 1 s when
combined with
[OpenBLAS#6105](https://github.com/OpenMathLib/OpenBLAS/pull/6105), which
removes the contention between threads calling BLAS concurrently, and will
ship in the next OpenBLAS release.

**Encoders.** Profiling a typical tabular pipeline showed `OneHotEncoder` and
`OrdinalEncoder` could take a surprising share of the time in some settings.
Itamar Turner-Trauring sped up unique value counts for strings
([#34386](https://github.com/scikit-learn/scikit-learn/pull/34386)). I made
the encoders work on F-ordered data
([#34392](https://github.com/scikit-learn/scikit-learn/pull/34392)). The next
step, [#34678](https://github.com/scikit-learn/scikit-learn/pull/34678), tries
to leverage the internal encoding of pandas categorical columns, but getting
every edge case to behave exactly as on `main` turned out to be complex, so
it's paused for now.

**Trees.** Binning is the main reason `sklearnex`'s random forests are faster,
and the next big item is to add it to scikit-learn's `RandomForest*` and
`ExtraTrees*`. In parallel, I'm experimenting with faster sorting for exact
splits: SIMD sort and radix sort. Radix sort gives very promising results on
the laptop, but less good ones on the server for now.

**Linear models.** Copies of `X` can take a large share of the fit time of
linear models: they are slow, single-threaded and double the peak memory,
while the rest of the fit runs in multi-threaded BLAS. So we're looking for
the copies that can be avoided.

**Benchmarks.**

- More estimators: `PCA`, `LogisticRegressionCV` and `RidgeCV`.
- XGBoost, LightGBM and CatBoost, to compare `HistGradientBoosting*` with them
  systematically.
- Discrete Intel GPUs, and more CUDA results to compare them with.
- A new version of the Phoronix scikit-learn workload, based on a subset of
  these cases, that we'd like to create in collaboration with the Phoronix
  maintainers.

**Threads and the Python array API.** These are the other workstreams of the
collaboration. Itamar Turner-Trauring (Quansight) and Olivier Grisel (Probabl)
work on how scikit-learn and its dependencies manage threads, in particular
for free-threaded Python:
[threadpoolctl 3.7.0](https://github.com/joblib/threadpoolctl/blob/master/CHANGES.md#370-2026-09-15)
can now set thread-local limits for MKL and OpenBLAS, loky detects physical
cores better on big machines, and the next joblib release will be
[thread-safe](https://github.com/joblib/joblib/pull/1834) and
[protect against nested oversubscription](https://github.com/joblib/joblib/pull/1825).
Python array API support also keeps progressing, with an Intel GPU CI in
[scikit-learn-intel-workflow](https://github.com/probabl-ai/scikit-learn-intel-workflow).

If you run scikit-learn on big machines, or see a result that doesn't match
your experience, please
[open an issue](https://github.com/probabl-ai/scikit-learn-benchmarks/issues).
I'm particularly curious about workloads where a single fit on many cores
matters to you: so far, most of our cases suggest that the outer loop is
where the cores should go.

## Acknowledgements

This work is funded by Intel as part of its collaboration with Probabl. The
infrastructure work on threading parallelism in scikit-learn is also
supported by NASA through the ROSES grant 80NSSC25K7215, "Ensuring a fast and
secure core for scientific Python".

Thanks to [Olivier Grisel](https://github.com/ogrisel) for the guidance and
the reviews, to [Itamar Turner-Trauring](https://github.com/itamarst) for the
encoders and threading work, and to the scikit-learn-intelex and oneDAL
maintainers at Intel for providing the hardware, and for their quick answers
and fixes, in particular [David Cortes](https://github.com/david-cortes-intel).
