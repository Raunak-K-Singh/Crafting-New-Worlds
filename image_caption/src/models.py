import torch.nn as nn
import torch.nn.functional as F

class AdaptiveFeatureFusion(nn.Module):
    """Adaptive feature fusion layer that combines features from multiple sources."""
    
    def __init__(self, input_dims: List[int], output_dim: int):
        super().__init__()
        self.fc_layers = nn.ModuleList([
            nn.Linear(dim, output_dim) for dim in input_dims
        ])
        self.output_layer = nn.Linear(len(input_dims) * output_dim, output_dim)

    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        fused_features = [layer(features[name]) for name, layer in zip(features.keys(), self.fc_layers)]
        concatenated_features = torch.cat(fused_features, dim=-1)
        return self.output_layer(concatenated_features)

class VisualStyleEncoder(nn.Module):
    """Encodes visual style features from images."""
    
    def __init__(self, config):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(128 * (config.image_size // 4) * (config.image_size // 4), 256)

    def forward(self, image):
        x = F.relu(self.conv1(image))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  
        return F.relu(self.fc(x))

class EmotionDetector(nn.Module):
    """Detects emotions from images using a simple CNN."""
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(32 * 62 * 62, 128)
        self.fc2 = nn.Linear(128, 5)  

    def forward(self, image):
        x = F.relu(self.conv1(image))
        x = self.pool(x)
        x = x.view(x.size(0), -1)  
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=-1)

class EnhancedMultiModalAttention(nn.Module):
    """Enhanced attention mechanism with relative position encoding and sparse attention."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.num_attention_heads = config.num_attention_heads or 12  
        self.attention_head_size = config.hidden_size // self.num_attention_heads
        
        self.query_layer = nn.Linear(config.hidden_size, self.num_attention_heads * self.attention_head_size)
        self.key_layer = nn.Linear(config.hidden_size, self.num_attention_heads * self.attention_head_size)
        self.value_layer = nn.Linear(config.hidden_size, self.num_attention_heads * self.attention_head_size)
        
        self.rel_pos_embedding = nn.Parameter(torch.randn(2 * config.max_caption_length - 1, self.attention_head_size))
        
        self.dropout_layer = nn.Dropout(0.1)
        self.output_layer = nn.Linear(self.num_attention_heads * self.attention_head_size, config.hidden_size)
        
        self.layer_norm = nn.LayerNorm(config.hidden_size)

    def compute_position_bias(self, hidden_states):
         seq_length=hidden_states.size(1) 
         position_biases=self.rel_pos_embedding[:seq_length,:seq_length] 
         return position_biases.unsqueeze(0).expand(hidden_states.size(0), -1 , -1)

     def forward(
         self,
         hidden_states,
         attention_mask=None,
         output_attentions=False,
     ):
         batch_size , seq_length , _=hidden_states.size()
         
         query_layer=self.query_layer(hidden_states).view(
             batch_size , seq_length ,self.num_attention_heads ,self.attention_head_size 
         ).transpose(1 ,2) 
         
         key_layer=self.key_layer(hidden_states).view(
             batch_size , seq_length ,self.num_attention_heads ,self.attention_head_size 
         ).transpose(1 ,2) 
         
         value_layer=self.value_layer(hidden_states).view(
             batch_size , seq_length ,self.num_attention_heads ,self.attention_head_size 
         ).transpose(1 ,2) 
         
         position_bias=self.compute_position_bias(hidden_states)
         
         attention_scores=torch.matmul(query_layer ,key_layer.transpose(-1 ,-2)) + position_bias
        
         attention_scores /= (self.attention_head_size **0.5)
         
         if attention_mask is not None:
             attention_scores += attention_mask
        
         attention_probs=F.softmax(attention_scores ,dim=-1)
        
         context_layer=torch.matmul(attention_probs ,value_layer).transpose(1 ,2).contiguous()
        
         context_layer=context_layer.view(batch_size ,-1) 
        
         attention_output=self.output_layer(context_layer)
         
         outputs=(attention_output + hidden_states) if output_attentions else (attention_output,)
        
         return outputs

class EnhancedImageEncoder(nn.Module):
   """Enhanced image encoder with multiple vision models and feature fusion."""
   
   def __init__(self ,config):
       super().__init__()
       
       # Load primary and backup models into a ModuleDict for easy access.
       self.vision_models=nn.ModuleDict()
       for model_name in config.model_configs["image_encoder"]:
           try:
               model_instance=self.load_vision_model(model_name) 
               self.vision_models[model_name]=model_instance 
           except Exception as e:
               logging.warning(f"Failed to load {model_name}: {e}")
       
       input_dims=[model.config.hidden_size for model in self.vision_models.values()]
       output_dim=2048 
       
       self.feature_fusion=AdaptiveFeatureFusion(input_dims=input_dims , output_dim=output_dim) 
       
       if config.enable_style_transfer:
           self.style_encoder=VisualStyleEncoder(config) 
       
       if config.enable_emotion_detection:
           self.emotion_detector=EmotionDetector() 

   def forward(self,image)->Dict[str , torch.Tensor]:
       """Forward pass through the enhanced image encoder."""
       
       features={}
       attention_maps={}
       
       for name , model in self.vision_models.items():
           try:
               feat=model(image)
               features[name]=feat 
           except Exception as e:
               logging.error(f"Error in {name}: {e}")
               continue

       fused_features=self.feature_fusion(features) 

       style_features=self.style_encoder(image) if hasattr(self,'style_encoder') else None 

       emotions=self.emotion_detector(image) if hasattr(self,'emotion_detector') else None 

       return {
           'features': fused_features,
           'style_features': style_features,
           'emotions': emotions 
       }

def load_vision_model(model_name:str)->nn.Module:
   """Load a vision model based on its name."""
   raise NotImplementedError("This function should load the specified vision model.")
