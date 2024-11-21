(* below are the activation Functions *)

(* this is the gelu activation *)
fun gelu x = 0.5 * x * (1.0 +Math.tanh (Math.sqrt(2.0 / Math.pi)* ( x + 0.044715 *Math.pow(x, 3.0))))

(* here is the softmax function *)
fun softmax(Vector v) =
    let
        val maxVal =foldl Real.max   (hd v) (tl v)
        val expVals = map(fn x => Math.exp(x - maxVal)) v
        val sumExp = foldl (op +)   0.0 expVals
    in
        Vector (map (fn x => x / sumExp)expVals)  end
