import copy
from torch.nn import CrossEntropyLoss

from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.roberta.modeling_roberta import (RobertaPreTrainedModel, 
    RobertaModel,
    RobertaClassificationHead
)

class RobertaForSequenceClassification(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]
    def __init__(self, model_name, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        
        de_config = copy.deepcopy(config)
        de_config.add_cross_attention = True
        de_config.is_decoder = True

        self.en_model = RobertaModel.from_pretrained(model_name, config=config)
        self.de_model = RobertaModel.from_pretrained(model_name, config=de_config)
        self.classifier = RobertaClassificationHead(config)

    def forward(
        self,
        input_ids=None,
        decoder_input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_outputs = self.en_model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        encoder_hidden_states = encoder_outputs[0]

        outputs = self.de_model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        sequence_outputs = outputs[0]
        logits = self.classifier(sequence_outputs)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )