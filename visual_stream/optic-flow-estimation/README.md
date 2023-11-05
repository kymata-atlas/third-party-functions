# Optic flow estimation

These functions are lightly adapted from Raudies' (2013[^raudies]) implementations[^raudies-code].  Included are the functions we have used for publications.

## Our changes

Our changes to Raudies code are minimal, primarily:

- Updating for compatibility with more recent versions of Matlab.
- Using Matlab's new `+dir` packaging syntax to allow functions to be dynamically loaded in code.
- Streaming of our particular stimuli rather than the example demo images included with Raudies' code.
- External access to "subfunctions" (i.e. internal state) of the top-level functions for `Heeger`.

## Versions

Two researchers worked on this prior to establishing proper distributed version-control practise, and as such there are two versions, the second being contained within `/andy-version`. The main version is good for use of the `HornSchunk` function, and the inner version is good for use of the `Heeger` functions.

We aim to find time to rationalise this in future.

## License

Everything herein is licensed under the GPL 3.0 (see `gpl-3.0.txt`).

[^raudies]: Raudies, F. (2013). [Optic flow](http://www.scholarpedia.org/article/Optic_flow). *Scholarpedia, 8*(7): 30724. doi: [10.4249/scholarpedia.30724](https://doi.org/10.4249/scholarpedia.30724).
[^raudies-code]: Cf. [Raudies' original code](https://github.com/fraudies/optic-flow-estimation).
