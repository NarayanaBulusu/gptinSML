(* below are the normalization Functions *)

(* this is layer normalization *)
fun layerNorm(Vector v) =
    let
      val mean = foldl (op +) 0.0   v / real (length v)
        val variance =  foldl (fn (x, acc)=> acc +Math.pow(x-mean, 2.0))0.0 v
        val stddev =Math.sqrt(variance)
    in
        Vector(map (fn x => (x -mean)/ stddev) v)
    end
