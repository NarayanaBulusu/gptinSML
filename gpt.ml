(* now, we are implementing a gpt model *)

(* this is a gpt model with multiple transformer blocks...*)
fun gptModel(input,config) =
    let
        fun applyBlocks input blocks =
            foldl(fn (block, acc)=> block acc)input blocks
        val transformerBlocks= replicate config 8 transformerBlock
    in
        applyBlocks(input, transformerBlocks)  (*note, foldl and map are built in*)
    end
