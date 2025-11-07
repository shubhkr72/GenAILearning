from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain_core.prompts import load_prompt
# it will download a model of around 2GB
llm = HuggingFacePipeline.from_model_id(
    model_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    task='text-generation',
    pipeline_kwargs=dict(
        temperature=1,
        # max_new_tokens=500
    )
)
model = ChatHuggingFace(llm=llm)

import streamlit as st

st.header('Research Tool')


paper_input = st.selectbox(
    "Select Research Paper Name",
    [
        # 🔹 Foundational ML
        "Gradient-Based Learning Applied to Document Recognition (LeCun et al., 1998)",
        "ImageNet Classification with Deep Convolutional Neural Networks (AlexNet, 2012)",
        "Very Deep Convolutional Networks for Large-Scale Image Recognition (VGG, 2014)",
        "Going Deeper with Convolutions (Inception/GoogLeNet, 2015)",
        "Deep Residual Learning for Image Recognition (ResNet, 2015)",
        "Densely Connected Convolutional Networks (DenseNet, 2016)",
        "Batch Normalization: Accelerating Deep Network Training (2015)",
        "Dropout: A Simple Way to Prevent Neural Networks from Overfitting (2014)",

        # 🔹 NLP / Transformers
        "Attention Is All You Need (Vaswani et al., 2017)",
        "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding (2018)",
        "GPT-2: Language Models are Unsupervised Multitask Learners (2019)",
        "GPT-3: Language Models are Few-Shot Learners (2020)",
        "T5: Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (2020)",
        "RoBERTa: A Robustly Optimized BERT Pretraining Approach (2019)",
        "DistilBERT: A Smaller, Faster, Cheaper Transformer (2019)",
        "LLaMA: Open and Efficient Foundation Language Models (2023)",

        # 🔹 Computer Vision
        "R-CNN: Rich Feature Hierarchies for Accurate Object Detection (2014)",
        "Fast R-CNN (2015)",
        "Faster R-CNN: Towards Real-Time Object Detection (2015)",
        "Mask R-CNN (2017)",
        "YOLO: You Only Look Once (2016)",
        "Vision Transformers (ViT, 2020)",
        "DETR: End-to-End Object Detection with Transformers (2020)",

        # 🔹 Generative Models
        "Generative Adversarial Nets (GANs, 2014)",
        "Conditional GANs (cGANs, 2014)",
        "CycleGAN: Unpaired Image-to-Image Translation (2017)",
        "StyleGAN: A Style-Based Generator Architecture for GANs (2019)",
        "Diffusion Models Beat GANs on Image Synthesis (2021)",
        "Denoising Diffusion Probabilistic Models (DDPM, 2020)",
        "Stable Diffusion (2022)",

        # 🔹 Reinforcement Learning
        "Playing Atari with Deep Reinforcement Learning (Deep Q-Learning, 2013)",
        "Mastering the Game of Go with Deep Neural Networks and Tree Search (AlphaGo, 2016)",
        "Proximal Policy Optimization Algorithms (PPO, 2017)",
        "Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm (AlphaZero, 2017)",
        "Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model (MuZero, 2019)",

        # 🔹 Diffusion & Multimodal
        "CLIP: Learning Transferable Visual Models from Natural Language Supervision (2021)",
        "DALL·E: Zero-Shot Text-to-Image Generation (2021)",
        "Flamingo: A Visual Language Model for Few-Shot Learning (2022)",
        "BLIP-2: Bootstrapped Language-Image Pretraining (2023)",
    ]
)


style_input = st.selectbox(
    "Select Explanation Style",
    [
        # 🔹 General Learning Levels
        "Beginner-Friendly",
        "Intermediate",
        "Advanced",
        "Expert-Level / Research-Oriented",

        # 🔹 Technical Depth
        "Technical",
        "Code-Oriented",
        "Mathematical / Theoretical",
        "Algorithmic Step-by-Step",

        # 🔹 Learning Modes
        "Analogy-Based (Simple Real-World Comparison)",
        "Example-Based (With Practical Scenarios)",
        "Visualization-Focused (Explain with Diagrams / Intuition)",
        "Summary / Key Points Only",

        # 🔹 Tone or Context
        "Conversational / Intuitive",
        "Formal Academic",
        "Interview Preparation Focused",
        "Implementation Guide (with Best Practices)",
        "Error Analysis & Debugging Focused",
        "Historical Context (Paper Motivation & Evolution)",
        "Concept to Code Walkthrough",
        "Research Paper Summary Format",
        "AI Instructor Mode (Teaching Step-by-Step)",
    ]
)


length_input = st.selectbox(
    "Select Explanation Length",
    [
        # 🔹 Concise Summaries
        "Very Short (1-2 sentences / TL;DR)",
        "Short (1-2 paragraphs)",
        
        # 🔹 Balanced Detail
        "Medium (3-5 paragraphs)",
        "Detailed (6-8 paragraphs)",
        
        # 🔹 In-Depth / Extended
        "Long (Comprehensive breakdown)",
        "Very Long (Research-level depth)",
        
        # 🔹 Alternate Formats
        "Bullet Points / Key Takeaways",
        "Step-by-Step Walkthrough",
        "Slide Summary (for Presentations)",
        "Executive Summary (for quick brief)",
    ]
)


# prompt template

# template = """
# You are an expert AI research explainer.

# Your task is to summarize and explain the research paper titled: **"{paper_input}"**.

# Please follow these user preferences:

# - **Explanation Style:** {style_input}  
# - **Explanation Length:** {length_input}

# ### Guidelines:

# 1. **Mathematical & Technical Details**
#    - Include important mathematical formulas or equations mentioned in the paper.
#    - When possible, illustrate key ideas with intuitive *pseudo-code* or *short code snippets*.
#    - Keep the technical level consistent with the chosen explanation style.

# 2. **Analogies & Intuition**
#    - Use simple, real-world analogies to clarify difficult or abstract concepts.
#    - Focus on helping the reader *understand the core idea*, not just the results.

# 3. **Accuracy & Completeness**
#    - Base the summary only on actual paper content.
#    - If some information is missing or unclear, write: **"Insufficient information available"** instead of guessing.

# Finally, ensure your summary is **clear, logically structured, and consistent** with the selected style and length.
# """

template=load_prompt("template.json")

prompt=template.invoke(
    {
        'paper_input':paper_input,
        'style_input':style_input,
        'length_input':length_input
    }
)
if st.button('Summarize'):
    result=model.invoke(prompt).content
    print(result)
    assistant_text = result.split("<|assistant|>")[-1].strip()
    print(assistant_text)

    st.text(assistant_text)