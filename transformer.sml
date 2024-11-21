(* this is aransformer Block *)

fun transformerBlock(input, weightsQ, weightsK, weightsV, config) =
    let
        (* theseare linear transformations forquery,key, and value *)
        val q = matmul(input, weightsQ) 
        val k =matmul(input, weightsK) 
        val v =matmul(input, weightsV)

        (* applying self-attention... *)
        val attnOutput = attention(q,k, v)

        (* red conn + layer normalization (forfirst layer) *)
        val residual1 =map (fn (x, y) =>x+ y) (ListPair.zip (hd input, hd attnOutput))
        val norm1 =layerNorm(Vector residual1)

        (* this is a Feedforward network *)
        val ffnOutput = gelu (hd residual1) (*this applies GELU activation to  output *)
        val residual2 = map (fn (x, y)=> x+y)(ListPair.zip (hd norm1, hd ffnOutput)) (* res connection *)
        val norm2= layerNorm (Vector residual2) (* now mormalize the output *)
    in
        Matrix [residual2] end
