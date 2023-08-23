//! This module defines the [`Destination`] type, which is used to safely manipulate callbacks.

use core::convert::{AsMut, AsRef};
use core::mem::MaybeUninit;
use std::cell::Cell;
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};

/// Destination<'a, R> represents a destination location in memory that must be filled with
/// an R. The lifetime 'a is the exact amount of time we have to fill in the value. We must
/// fill the value in before 'a ends, and we have access to the location until 'a ends.
/// Consequently, Destination<'a, R> is invariant with respect to 'a. The preferred way of
/// creating a member of this type is with [`Destination::with_destination`].
///
/// Note that [`Destination<'a, R>`] is representationally equivalent to a pointer.
#[repr(transparent)]
pub struct Destination<'a, R> {
    dest: WithInvariant<'a, &'a mut MaybeUninit<R>>,
}

/// A datum of type Finished<'a> can only be constructed from a [`Destination<'a, R>`]. It
/// represents a proof that we actually did fill in the value. This type has zero size and
/// has alignment 1.
///
/// I claim that Finished should be FFI-safe when used as the return type; the corresponding
/// return type in C should be void. It is unclear whether this is actually guaranteed, but as of
/// the current version of the compiler (1.71.1), the compiler does make this true. For now,
/// we must use #\[allow(improper_ctypes_definition)\] when defining an `extern "C"` function
/// that returns a Finished.
#[repr(transparent)]
pub struct Finished<'a> {
    // Finished must be invariant with respect to 'a.
    _phantom: WithInvariant<'a, ()>,
}

/// repr(transparent) equivalent to T, invariant with respect to lifetime 'a.
#[repr(transparent)]
struct WithInvariant<'a, T> {
    value: T,
    _phantom: PhantomData<Cell<&'a i32>>,
}

impl<'a, T> WithInvariant<'a, T> {
    pub fn new(value: T) -> Self {
        Self {
            value,
            _phantom: PhantomData,
        }
    }

    pub fn into(self) -> T {
        self.value
    }
}

impl<'a, T> Deref for WithInvariant<'a, T> {
    type Target = T;

    fn deref(&self) -> &T {
        &self.value
    }
}

impl<'a, T> DerefMut for WithInvariant<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.value
    }
}

impl<'a> Finished<'a> {
    fn new() -> Self {
        Self {
            _phantom: WithInvariant::new(()),
        }
    }
}

impl<'a, R> AsRef<MaybeUninit<R>> for Destination<'a, R> {
    fn as_ref(&self) -> &MaybeUninit<R> {
        &*self.dest
    }
}

impl<'a, R> AsMut<MaybeUninit<R>> for Destination<'a, R> {
    fn as_mut(&mut self) -> &mut MaybeUninit<R> {
        &mut *self.dest
    }
}

impl<'a, R> Destination<'a, R> {
    /// Make a new destination. It's dangerous to call this, since we might accidentally create
    /// two destinations that have the same lifetime.
    fn new(dest: &'a mut MaybeUninit<R>) -> Self {
        Self {
            dest: WithInvariant::new(dest),
        }
    }

    /// One way to prove to the compiler that we truly did fill the destination is to actually
    /// fill it. Once we've filled in the destination, we can keep around a mutable reference
    /// to it if we wish.
    ///
    /// # Example
    ///
    /// ```
    /// use libffi::destination::Destination;
    ///
    /// let result: i32 = Destination::with_destination(|destination| {
    ///     let (destination, finished) = destination.initialize(0);
    ///     for x in 1..=100 {
    ///         *destination += x;
    ///     }
    ///     finished
    /// });
    ///
    /// assert_eq!(result, 5050);
    /// ```
    pub fn initialize(self, value: R) -> (&'a mut R, Finished<'a>) {
        let dest = self.dest.into().write(value);
        (dest, Finished::new())
    }

    /// Just like [`Destination::initialize`], except it throws away the reference to the
    /// initialized value.
    ///
    /// # Example
    ///
    /// ```
    /// use libffi::destination::Destination;
    /// let result = Destination::with_destination(|d| {
    ///     d.finish(5)
    /// });
    ///
    /// assert_eq!(5, result);
    /// ```
    ///
    pub fn finish(self, value: R) -> Finished<'a> {
        self.initialize(value).1
    }

    /// If you have fully initialized the destination through the [`AsMut`] trait, but the
    /// compiler cannot verify this, you can use this method to assert to the compiler that
    /// the destination is initialized.
    ///
    /// # Safety
    /// You as the author are required to fully initialize the `R` before calling this method.
    /// If you don't, immediate undefined behavior ensues, as the output `&mut R` refers to a
    /// value that hasn't been initialized.
    ///
    /// # Example
    ///
    /// ```
    /// use std::mem::MaybeUninit;
    /// use libffi::destination::Destination;
    ///
    /// let array: [i32; 3] = Destination::with_destination(|mut destination| {
    ///     let mut uninit_reference: &mut MaybeUninit<[i32; 3]> = destination.as_mut();
    ///     {
    ///         let mut reference_uninit: &mut [MaybeUninit<i32>; 3] = unsafe {
    ///             &mut*(uninit_reference as *mut _ as *mut _)
    ///         };
    ///         for i in 0..3 {
    ///             let output = i as i32;
    ///             reference_uninit[i].write(output);
    ///         }
    ///     }
    ///     let (init_reference, finished) = unsafe { destination.initialized_unchecked() };
    ///     for i in 0..3 {
    ///        init_reference[i] += 1;
    ///     }
    ///     finished
    /// });
    ///
    /// for i in 0..3 {
    ///     assert_eq!(array[i], (i as i32) + 1);
    /// }
    /// ```
    pub unsafe fn initialized_unchecked(self) -> (&'a mut R, Finished<'a>) {
        (self.dest.into().assume_init_mut(), Finished::new())
    }

    /// Just like [`Destination::initialized_unchecked`], except we throw away the reference to the
    /// initialized value.
    ///
    /// # Safety
    ///
    /// Same requirements as [`Destination::initialized_unchecked`]. You must have initialized
    /// the `R` at the destination, or you'll see immediate undefined behavior.
    ///
    /// # Example
    ///
    /// ```
    /// use std::mem::MaybeUninit;
    /// use libffi::destination::Destination;
    ///
    /// let array: [i32; 3] = Destination::with_destination(|mut destination| {
    ///     let mut uninit_reference: &mut MaybeUninit<[i32; 3]> = destination.as_mut();
    ///     {
    ///         let mut reference_uninit: &mut [MaybeUninit<i32>; 3] = unsafe {
    ///             &mut*(uninit_reference as *mut _ as *mut _)
    ///         };
    ///         for i in 0..3 {
    ///             let output = 1 + i as i32;
    ///             reference_uninit[i].write(output);
    ///         }
    ///     }
    ///     unsafe { destination.finish_unchecked() }
    /// });
    ///
    /// for i in 0..3 {
    ///     assert_eq!(array[i], (i as i32) + 1);
    /// }
    /// ```
    pub unsafe fn finish_unchecked(self) -> Finished<'a> {
        self.initialized_unchecked().1
    }

    /// We create a new Destination to store a value of type `R`. We then invoke the function
    /// `f`, passed as an input, on this destination. Because of `f`'s polymorphic nature, it is
    /// guaranteed to actually fill the destination it gets passed. We then output the contents
    /// of this destination.
    ///
    /// # Examples
    ///
    /// ```
    /// use libffi::destination::Destination;
    ///
    /// let x = 5;
    ///
    /// assert_eq!(5, Destination::with_destination(|d| d.initialize(x).1));
    ///
    /// let z = 47;
    ///
    /// assert_eq!(52, Destination::with_destination(|d| {
    ///     let (result, finished) = d.initialize(x);
    ///     *result += z;
    ///     finished
    /// }));
    /// ```
    pub fn with_destination(f: impl for<'b> FnOnce(Destination<'b, R>) -> Finished<'b>) -> R {
        let mut result = MaybeUninit::uninit();
        f(Destination::new(&mut result));

        // SAFETY: We made it to this point, so we know that the call to f terminated. Therefore,
        // we produced a valid value of type Finished<'b>. Because f is polymorphic in 'b, the only
        // way the compiler could prove that the output has lifetime 'b is by proving the output
        // came from the input. Thus, we must have called `finished_unchecked` or `finished` on
        // the input, which means the input is indeed initialized.
        unsafe { result.assume_init() }
    }
}
