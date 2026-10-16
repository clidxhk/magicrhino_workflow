from openai import OpenAI
import base64
import httpx

class ExpertChat:
    def __init__(self, api_key, base_url, model, temperature):
        # OpenAI 1.52.x passes the removed `proxies` argument when it creates
        # its own client with httpx 0.28.x. Supplying a client avoids that
        # incompatible constructor path while retaining environment proxies.
        self.http_client = httpx.Client(
            timeout=httpx.Timeout(60.0, connect=15.0),
            follow_redirects=True,
        )
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=self.http_client,
        )
        self.model = model
        self.temperature = temperature

    def close(self):
        self.client.close()
    
    def generate_text_analysis(self, metrics_data):
        """生成文本分析"""
        text_content = "\n".join([f"{key}: {value}" for key, value in metrics_data.items()])
        
        system_prompt = """# [Role Background]: You are an experienced AI expert who focuses on the evaluation and improvement of machine learning models. You are familiar with various machine learning metrics and their significance in different application scenarios. You can judge the performance of the model based on these metrics and provide optimization suggestions. You communicate with the team in a clear, concise and professional manner to help them improve the effectiveness of the model. # [Task Overview]: Your task is to evaluate the advantages and disadvantages of the model based on the performance metrics of the provided machine learning model and make improvement suggestions. You need to explain the meaning of each metric, point out the advantages and disadvantages of the model, and give specific optimization strategies. # [Output Format]:1. **Model Evaluation** - **Metric Explanation**: Explain in detail each provided machine learning metric and its significance. - **Model Performance**: Evaluate the performance of the model based on the metric and point out its advantages and disadvantages. 2. **Optimization Suggestions** - **Data Processing**: Make suggestions related to data preprocessing and feature engineering. 3. **Summary** - **Overall Evaluation**: Summarize the overall performance of the model. - **Next Action**: Propose the next action plan. # [Task Requirements] 1. **Comprehensiveness**: The evaluation should cover all provided performance indicators to ensure a comprehensive analysis. 2. **Professionalism**: Use professional terms and clear explanations to ensure that the team can understand your evaluation and suggestions. 3. **Practicality**: The proposed optimization suggestions should be operational and can be implemented in actual projects. 4. **Simplicity**: Keep the language concise and clear, and avoid lengthy descriptions. 5. Must be output in Markdown format. 6. Only reply about 200 words"""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text_content},
            ],
            max_tokens=2048,
            n=1,
            temperature=self.temperature,
        )
        return response.choices[0].message.content
    
    def generate_image_analysis(self, base64_image, user_text):
        """生成图像分析"""
        system_prompt = """
        **MagicRhino :**
                        
        [Role Background]: 
                        
            1. You are a SHAP (SHapley Additive exPlanations) model explanation expert, 
                specializing in using the SHAP to interpret machine learning models. 
            2. You possess deep theoretical knowledge and extensive practical experience, 
                enabling you to clearly explain the reasons behind model predictions and feature contributions; 

        [Task Overview]: 
                        
            1. Your task is to help users understand the SHAP analysis results by image. 
            2. You need to explain the contribution of features; 
                        
        [Task Requirements]: 
                        
            1. Clarity and Accuracy: Explanations should be clear and accurate. 
            2. Structured: Answers should be structured for easy understanding and reference by users."
        """
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": "low"
                            }
                        }
                    ]
                }
            ],
            max_tokens=1000,
            n=1,
            temperature=self.temperature,
        )
        return response.choices[0].message.content

def encode_image_to_base64(uploaded_file):
    """将上传的文件编码为base64"""
    return base64.b64encode(uploaded_file.read()).decode('utf-8')

def get_default_image_prompt():
    """获取默认的图像分析提示文本"""
    return """
**MagicRhino :**
           
[Role Background]: 
           
    1. You are a SHAP (SHapley Additive exPlanations) model explanation expert, 
        specializing in using the SHAP to interpret machine learning models. 
    2. You possess deep theoretical knowledge and extensive practical experience,             
        enabling you to clearly explain the reasons behind model predictions and feature contributions; 

[Task Overview]: 
        
    1. Your task is to help users understand the SHAP analysis results by image. 
    2. You need to explain the contribution of features; 
           
[Task Requirements]: 
           
    1. Clarity and Accuracy: Explanations should be clear and accurate. 
    2. Structured: Answers should be structured for easy understanding and reference by users."
"""