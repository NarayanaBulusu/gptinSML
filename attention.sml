(*now this is the self-attention process that is emphasized whilst checking scores and weights and ensuring accuracy... *)

(*attention function *)
fun attention(q, k, v) =
    let
        val scores =matmul(q,transpose k)
        val Vector weights = softmax(Vector (hd (hd scores)))
    in
        matmul(Matrix [weights],v)
    end
