//! This module defines the [`Destination`] type, which is used to safely manipulate callbacks.

use core::convert::{AsMut, AsRef};
use core::mem::MaybeUninit;
use std::cell::Cell;
use std::marker::PhantomData;

/// A zero-sized-type which is invariant with respect to 'a.
type Invariant<'a> = PhantomData<Cell<&'a i32>>;

/// Destination<'a, R> represents a destination location in memory that must be filled with
/// an R. The lifetime 'a is the exact amount of time we have to fill in the value. We must
/// fill the value in before 'a ends, and we have access to the location until 'a ends.
/// Consequently, Destination<'a, R> is invariant with respect to 'a. The preferred way of
/// creating a member of this type is with [`Destination::with_destination`].
///
/// Note that [`Destination<'a, R>`] is representationally equivalent to a pointer.
#[repr(transparent)]
pub struct Destination<'a, R> {
    dest: &'a mut MaybeUninit<R>,
    _phantom: Invariant<'a>,
}

/// A datum of type Finished<'a> can only be constructed from a [`Destination<'a, R>`]. It
/// represents a proof that we actually did fill in the value. This type has zero size and
/// has alignment 1.
#[repr(transparent)]
pub struct Finished<'a> {
    // Finished must be invariant with respect to 'a.
    _phantom: Invariant<'a>,
}

impl<'a> Finished<'a> {

    fn new() -> Self {
        Self {
            _phantom: PhantomData,
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
            dest,
            _phantom: PhantomData,
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
    ///     let (destination, finished) = destination.finished(0);
    ///     for x in 1..=100 {
    ///         *destination += x;
    ///     }
    ///     finished
    /// });
    ///
    /// assert_eq!(result, 5050);
    /// ```
    pub fn finished(self, value: R) -> (&'a mut R, Finished<'a>) {
        let dest = self.dest.write(value);
        (dest, Finished::new())
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
    ///         let mut reference_uninit: &mut [MaybeUninit<i32>; 3] = unsafe { &mut*(uninit_reference as *mut _ as *mut _)};
    ///         for i in 0..3 {
    ///             let output = (i as i32) + 1;
    ///             reference_uninit[i].write(output);
    ///         }
    ///     }
    ///     unsafe { destination.finished_unchecked() }.1
    /// });
    ///
    /// for i in 0..3 {
    ///     assert_eq!(array[i], (i as i32) + 1);
    /// }
    /// ```
    pub unsafe fn finished_unchecked(self) -> (&'a mut R, Finished<'a>) {

        (self.dest.assume_init_mut(), Finished::new())
    }

    /// We create a new Destination to store a value of type `R`. We then invoke the function
    /// `f`, passed as an input, on this destination. Because of `f`'s polymorphic nature, it is
    /// guaranteed to actually fill the destination it gets passed. Therefore,
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
