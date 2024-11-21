(* main file to execute the gpt *)

(* here we import all the modules we use*)
use "tensor.sml";
use "activations.sml";
use "normalization.sml";
use "attention.sml";
use "transformer.sml";
use "gpt.sml";
use "train.sml";

(*config*)
val config = 128
val input = Matrix [[1.0, 2.0, 3.0]]

(*training....*)
val trainedModel = train gptModel input config
