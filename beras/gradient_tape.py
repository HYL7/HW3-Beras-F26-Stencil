from collections import defaultdict

from beras.core import Diffable, Tensor

class GradientTape:

    def __init__(self):
        # Dictionary mapping the object id of an output Tensor to the Diffable layer it was produced from.
        self.previous_layers: defaultdict[int, Diffable | None] = defaultdict(lambda: None)

    def __enter__(self):
        # When tape scope is entered, all Diffables will point to this tape.
        if Diffable.gradient_tape is not None:
            raise RuntimeError("Cannot nest gradient tape scopes.")

        Diffable.gradient_tape = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # When tape scope is exited, all Diffables will no longer point to this tape.
        Diffable.gradient_tape = None

    def gradient(self, target: Tensor, sources: list[Tensor]) -> list[Tensor]:
        """
        Computes the gradient of the target tensor with respect to the sources.

        :param target: the tensor to compute the gradient of, typically loss output
        :param sources: the list of tensors to compute the gradient with respect to
        In order to use tensors as keys to the dictionary, use the python built-in ID function here: https://docs.python.org/3/library/functions.html#id.
        """

        ### TODO: Populate the grads dictionary with {weight_id, weight_gradient} pairs and return the gradients of the sources

        queue = [target]                    ## Live queue; will be used to propagate backwards via breadth-first-search.
        grads = defaultdict(lambda: None)   ## Grads to be recorded. Initialize to None. Note: stores {id: list[gradients]}
        # Use id(tensor) to get the object id of a tensor object.
        # in the end, your grads dictionary should have the following structure:
        # {id(tensor): [gradient]}

        # What tensor and what gradient is for you to implement!
        # compose_input_gradients and compose_weight_gradients are methods that will be helpful

        while queue != []:
            current = queue.pop(0)
            upper_grad = grads[id(current)]
            layer = self.previous_layers[id(current)]
            if layer is not None:
                inputs = layer.inputs
                weights = layer.weights
                inputs_grad = layer.compose_input_gradients(upper_grad)
                weights_grad = layer.compose_weight_gradients(upper_grad)
                for i in range (len(inputs)):
                    if grads[id(inputs[i])] is None:
                        grads[id(inputs[i])] = [inputs_grad[i]]
                    else:
                       grads[id(inputs[i])] = [grads[id(inputs[i])][0] + inputs_grad[i]]
                    if id(inputs[i]) not in [id(tensor) for tensor in queue]:
                        queue.append(inputs[i])
                for w in range (len(weights)):
                    if grads[id(weights[w])] is None:
                        grads[id(weights[w])] = [weights_grad[w]]
                    else:
                        grads[id(weights[w])] = [grads[id(weights[w])][0] + weights_grad[w]]

        output = []
        for s in sources:
            output.append(grads[id(s)][0])

        return output