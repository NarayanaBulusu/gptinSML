(*here we have the operations for tensor *)

datatype Tensor = Vector of real list | Matrix of real list list

(* dot product for vectors *)
fun dot (Vector v1, Vector v2) =
    foldl (op +) 0.0   (  ListPair.map (op *) (v1, v2))

(* transpose for matrices *)
fun transpose (Matrix m) =
    let
    (*col j is....*)
        fun col j = map (  fn row => List.nth(row, j))   m
    in
        Matrix(List.tabulate(length  (hd m), col))
    end

(* matrix multiplication   *)
fun matmul (Matrix m1, Matrix m2) =
    let
        val Matrix m2T =    transpose (Matrix m2)
        fun rowCol row col = dot (Vector row, Vector col)
    in
        Matrix (map (fn r =>map    (fn c => rowCol r c) m2T) m1)
    end
