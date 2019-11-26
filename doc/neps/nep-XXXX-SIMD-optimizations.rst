===============================================================
NEP XXXX — Using SIMD optimization instructions for performance
===============================================================

:Author: Sayid Adel, Matti Picus
:Status: Draft
:Type: Standards
:Created: 2019-11-25
:Resolution: none


Abstract
--------

While compilers are getting better at using hardware-specific routines to
optimize code, they sometimes do not produce optimal results. Also, we would
like to be able to copy binary c-extension modules from one machine to another
without recompiling.

We have a mechanism in the ufunc machinery to choose optimal loops. This NEP
proposes a mechanism to build on that for many more features and architectures.
The steps would be:
- Establish a baseline of CPU features for minimal support
- Write explicit code to take advantage of well-defined, architecture-agnostic,
  universal intrisics which capture features available across architectures.
- Capture those universal intrisics in a set of C macros that at compile time
  would build code paths for each feature from the baseline up to the maximum
  set of features available on that architecture.
- At runtime, discover which CPU features are available, and choose from among
  the possible code paths accordingly.

Motivation and Scope
--------------------

Traditionally NumPy has counted on the compilers to generate optimal code.
Recently there were discussions around whether this is `good enough`_ or
if hand-written code is needed. Some architecture-specific code was added to
NumPy for `fast routines`_ on x86 in ufuncs, using the loop-resolution routines
to choose the correct loop for the architecture. However the code is not
generic and does not generalize to other architectures. It would be nice if
the universal intrinsics would be available to other libraries like SciPy or
astropy whoe also build ufuncs, but that is not an explicit goal of the first
implementation of this NEP.

Usage and Impact
----------------

This section describes how users of NumPy will use features described in this
NEP. It should be comprised mainly of code examples that wouldn't be possible
without acceptance and implementation of this NEP, as well as the impact the
proposed changes would have on the ecosystem. This section should be written
from the perspective of the users of NumPy, and the benefits it will provide
them; and as such, it should include implementation details only if
necessary to explain the functionality.

Backward compatibility
----------------------

This section describes the ways in which the NEP breaks backward compatibility.

The mailing list post will contain the NEP up to and including this section.
Its purpose is to provide a high-level summary to users who are not interested
in detailed technical discussion, but may have opinions around, e.g., usage and
impact.

Detailed description
--------------------

This section should provide a detailed description of the proposed change.
It should include examples of how the new functionality would be used,
intended use-cases and pseudo-code illustrating its use.


Related Work
------------

This section should list relevant and/or similar technologies, possibly in other
libraries. It does not need to be comprehensive, just list the major examples of
prior and relevant art.


Implementation
--------------

This section lists the major steps required to implement the NEP.  Where
possible, it should be noted where one step is dependent on another, and which
steps may be optionally omitted.  Where it makes sense, each step should
include a link to related pull requests as the implementation progresses.

Any pull requests or development branches containing work on this NEP should
be linked to from here.  (A NEP does not need to be implemented in a single
pull request if it makes sense to implement it in discrete phases).


Alternatives
------------

If there were any alternative solutions to solving the same problem, they should
be discussed here, along with a justification for the chosen approach.


Discussion
----------

This section may just be a bullet list including links to any discussions
regarding the NEP:

- This includes links to mailing list threads or relevant GitHub issues.


References and Footnotes
------------------------

.. _`good enough`: https://github.com/numpy/numpy/pull/11113
.. _`fast routines`: https://github.com/numpy/numpy/pulls?q=is%3Apr+avx512+is%3Aclosed

.. [1] Each NEP must either be explicitly labeled as placed in the public domain (see
   this NEP as an example) or licensed under the `Open Publication License`_.

.. _Open Publication License: https://www.opencontent.org/openpub/


Copyright
---------

This document has been placed in the public domain. [1]_
