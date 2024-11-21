(* mse loss func(calculates how far predictions are from targets) *)
fun mseLoss (preds, targs) =
    let
        (* find the squared diffs for each pair of prediction and target *)
        val diffs =ListPair.map (fn (p, t)=> Math.pow(p-t,2.0)) (preds, targs)
        (* add all the squared diffs *)
        val total = oldl (op +) 0.0 diffs
        (* average the diffs to get the mean squared error *)
        val mean =total/real (length diffs)
    in
        mean end

(* gradient calculation u-- note this is an approximation... *)
fun computeGrads (mdl, inp, targ, eps) =
    let
        (* this helper tweaks one input param at a time *)
        fun perturbAndCalc i =
            let
                (*change the i-th input slightly by adding epsilon *)
                val perturbedInputs =map (fn (j, v)=> if i=j then v+ eps else v) 
                                      (ListPair.zip(List.tabulate(length inp, fn x=>x), inp))
                (* get predictions for both original and perturbed inputs *)
                val perturbedPred =mdl perturbedInputs
                val lossOrig = mseLoss(mdl inp, targ)
                val lossPert =mseLoss (perturbedPred, targ)
            in
                (* compute gradient as the rate of change in loss *)
                (lossPert-lossOrig)/eps
            end
    in
        (* retu rns a list of gradients for each input param *)
        map perturbAndCalc(List.tabulate(length inp, fn x=>x))
    end;

(* this is a training loop tp optimize model...check oncemoe and trace w example before committing... *)
fun train mdl data cfg =
    let
        (* set learning rate and epsilon for gradients *)
        val lr =0.01
        val eps =1e-5

        (* recursive fn for looping through epochs *)
        fun loop epoch mdl =
            if epoch>cfg then mdl (* stop training after cfg epochs *)
            else
                let
                    (* pick the first data batch (just a placeholder for now) *)
                    val (inputs, targets) =hd data
                    
                    (* make predictions using the model *)
                    val preds=mdl inputs
                    
                    (* calculate the loss (how bad the preds are) *)
                    val loss = mseLoss (preds, targets)
                    
                    (* find gradients of the loss w.r.t. the inputs *)
                    val grads = computeGrads (mdl, inputs, targets, eps)
                    
                    (* update model params with gradient descent *)
                    val updatedModel = map (fn (param, grad)=> param-lr * grad) 
                                       (ListPair.zip(mdl, grads))

                    (* print out progress after each epoch *)
                    val _ = print ("epoch: " ^ Int.toString epoch ^ ", loss: " ^ Real.toString loss ^ "\n")
                in
                    (* keep training with the updated model *)
                    loop (epoch + 1) updatedModel
                end
    in
        (*start training from epoch 1 *)
        loop 1 mdl
    end
