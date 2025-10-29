# Changelog

All notable changes to the Golomb Ruler Optimization project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.2.5](https://github.com/Gotman08/golomb/compare/v1.2.4...v1.2.5) (2025-10-29)


### Bug Fixes

* **style:** move inline comments to separate lines for constants ([05cdec3](https://github.com/Gotman08/golomb/commit/05cdec305b7addff8bb40a09f5a1eec66435a9a8))

## [1.2.4](https://github.com/Gotman08/golomb/compare/v1.2.3...v1.2.4) (2025-10-29)


### Bug Fixes

* **style:** use 2 spaces before inline comments ([66503e7](https://github.com/Gotman08/golomb/commit/66503e72f0884592aae846a87a443b803998cb52))

## [1.2.3](https://github.com/Gotman08/golomb/compare/v1.2.2...v1.2.3) (2025-10-29)


### Bug Fixes

* **style:** correct clang-format spacing violations ([de013e1](https://github.com/Gotman08/golomb/commit/de013e18254d1ded57e8ad0303e3c0357cee7482))

## [1.2.2](https://github.com/Gotman08/golomb/compare/v1.2.1...v1.2.2) (2025-10-29)


### Bug Fixes

* **style:** apply clang-format spacing rules ([b91996d](https://github.com/Gotman08/golomb/commit/b91996d56e351f43d8a27fa6420526ffeddb4934))

## [1.2.1](https://github.com/Gotman08/golomb/compare/v1.2.0...v1.2.1) (2025-10-29)


### Code Refactoring

* improve code quality and remove Windows remnants ([8d62f94](https://github.com/Gotman08/golomb/commit/8d62f943523e2d81b4ad29a7b2a6be9865192fc2))

## [1.2.0](https://github.com/Gotman08/golomb/compare/v1.1.3...v1.2.0) (2025-10-29)


### Features

* **exact:** integrate OR-Tools CP-SAT solver for exact solving ([c1900db](https://github.com/Gotman08/golomb/commit/c1900dbcadc65777de94f9b8cf0c0ea5c06065b4))


### Bug Fixes

* **ci:** add OR-Tools pre-compiled binary installation to release workflow ([0cfe59b](https://github.com/Gotman08/golomb/commit/0cfe59b701b69aaf6ab87e9a744d4824b1253e51))
* **ci:** correct OR-Tools extraction path on Windows ([7de1da3](https://github.com/Gotman08/golomb/commit/7de1da3dddf7d1fffda631b4d65b6567732a172b))
* **ci:** correct OR-Tools release version to v9.10.4067 ([fc0a507](https://github.com/Gotman08/golomb/commit/fc0a507c3ae4c12feab3975d59546c2247248a35))
* **ci:** use pre-compiled OR-Tools binaries to avoid timeout ([b220d7e](https://github.com/Gotman08/golomb/commit/b220d7ef002c916f81bd454751134f12b728ab1e))
* **cmake:** disable sanitizers when using pre-compiled OR-Tools ([6e8b1c5](https://github.com/Gotman08/golomb/commit/6e8b1c5af5c8e2891ecde1199580b65e37e76fea))
* **cmake:** skip pre-compiled OR-Tools for Windows Debug builds ([774fc30](https://github.com/Gotman08/golomb/commit/774fc3068bc13e30421d72739abf66e94df7347d))
* **exact:** add using declaration for operations_research::Domain ([9f76892](https://github.com/Gotman08/golomb/commit/9f76892a8641c7f9bd6c976767556d2a1007071b))
* **exact:** store CpModelProto before passing to solver ([60079fa](https://github.com/Gotman08/golomb/commit/60079faee8e0449be14e11e042eb11a1ab09c460))
* **exact:** use explicit using declarations for all OR-Tools types ([f3fc967](https://github.com/Gotman08/golomb/commit/f3fc967731064abfc8226fab68bf7609d3f6bd86))
* **exact:** use SolveWithParameters function instead of CpSolver class ([8491e1f](https://github.com/Gotman08/golomb/commit/8491e1f1983d0dac35d776bd70d8184aa6bfc9a5))
* **tests:** accept multiple optimal solutions for order 6 Golomb ruler ([2ee6a4b](https://github.com/Gotman08/golomb/commit/2ee6a4b2dd4c7e1fc7ecd8fed018ac6a155534c5))
* **tests:** wrap OR expression in parentheses for Catch2 ([cd852d4](https://github.com/Gotman08/golomb/commit/cd852d42797ec9806ac0ecc10c9fe223171bccd8))


### Documentation

* remove all Windows references from project ([85a8f8e](https://github.com/Gotman08/golomb/commit/85a8f8e5125dd44ac378832277789f0d09a52eea))


### Code Refactoring

* extract helper functions to reduce code duplication ([1cab3be](https://github.com/Gotman08/golomb/commit/1cab3be79de8f96cbc355eeb1f3c81eee8d36e75))

## [1.1.3](https://github.com/Gotman08/golomb/compare/v1.1.2...v1.1.3) (2025-10-28)


### Bug Fixes

* **windows:** resolve MSVC compilation errors ([e9107ac](https://github.com/Gotman08/golomb/commit/e9107acde849ea79f1e2185960661003d92e9519))

## [1.1.2](https://github.com/Gotman08/golomb/compare/v1.1.1...v1.1.2) (2025-10-28)


### Bug Fixes

* **nn:** implement default constructor as valid scalar tensor ([e332b15](https://github.com/Gotman08/golomb/commit/e332b1558ed44fa9b0f5bbb6b814f3a29ebbe792))
* **tests:** correct variable scope in zero_grad test ([8f8273d](https://github.com/Gotman08/golomb/commit/8f8273d0fdada2807a7935620f93c09682972b34))
* **tests:** prevent undefined behavior with temporary vectors ([41997c0](https://github.com/Gotman08/golomb/commit/41997c0ac3e31ce3d5676855c5f5efe1a903d1ad))
* **tests:** resolve Tensor constructor ambiguity ([1dbf5f1](https://github.com/Gotman08/golomb/commit/1dbf5f1c220cd0fadf6d52f844bff2cd9d9c7fa8))

## [1.1.1](https://github.com/Gotman08/golomb/compare/v1.1.0...v1.1.1) (2025-10-28)


### Bug Fixes

* **examples:** add missing cmath header for std::sqrt ([f7db226](https://github.com/Gotman08/golomb/commit/f7db22651a6c7d2c90a3b7bf006be82e7eaf4abd))

## [1.1.0](https://github.com/Gotman08/golomb/compare/v1.0.1...v1.1.0) (2025-10-28)


### Features

* **mcts:** integrate neural network for policy priors and value estimation ([c10b78d](https://github.com/Gotman08/golomb/commit/c10b78d3540edd54ecc5e1126cd0b45bbbc2b247))
* **nn:** add GolombNet with dual-head architecture (AlphaGo-style) ([2a1b645](https://github.com/Gotman08/golomb/commit/2a1b645aeac595f5a647480d2d14d6fe864223eb))
* **nn:** add linear layer with forward and backward passes ([e86a568](https://github.com/Gotman08/golomb/commit/e86a56861380deec970efc09c0da15b39d00aa97))
* **nn:** add state encoder for Golomb ruler representation ([ac4fdb7](https://github.com/Gotman08/golomb/commit/ac4fdb7268e33505c34b6eed99f59a6d225dc4c1))
* **nn:** add tensor class with matrix operations and activation functions ([647965c](https://github.com/Gotman08/golomb/commit/647965cd9a2c8bfc1e6cead2a489639bcd495dcb))


### Bug Fixes

* **build:** add missing golomb_nn dependency in benchmarks ([011fcee](https://github.com/Gotman08/golomb/commit/011fcee39d0015c37f094cd951026fa92349abdc))
* **nn:** add default constructor to Tensor class ([8c7e235](https://github.com/Gotman08/golomb/commit/8c7e235b0071b14ba1b5ab2aaab8cfee48284f02))


### Documentation

* **examples:** add demo program and testing guide ([e9e61dd](https://github.com/Gotman08/golomb/commit/e9e61dd634c1c40499c27997965a425f60825010))

## [1.0.1](https://github.com/Gotman08/golomb/compare/v1.0.0...v1.0.1) (2025-10-26)


### Bug Fixes

* **ci:** correct benchmark executable path ([6c09b6e](https://github.com/Gotman08/golomb/commit/6c09b6e6127468e0ae9325efdb347b4c8a80f11e))

## 1.0.0 (2025-10-26)


### Features

* add MCP integration, automated benchmarks and semantic release ([d428430](https://github.com/Gotman08/golomb/commit/d428430969152713d4f12467c449eb29525dd49f))


### Bug Fixes

* **ci:** correct Catch2 module path for v3.5.0 ([e615e76](https://github.com/Gotman08/golomb/commit/e615e76e1145e5f192cc36854c2c02805743ee74))
* **ci:** use correct Catch2 CMake variable name ([a62c5cb](https://github.com/Gotman08/golomb/commit/a62c5cbe79c8f3cc83ea24a04e59d5fa0dbe9beb))
* **mcts:** add missing algorithm header for std::find ([7221459](https://github.com/Gotman08/golomb/commit/722145968f723110f7a6fd20029a6dfb4ab6993a))

## [Unreleased]

Initial release pending. The changelog will be automatically generated by semantic-release.
