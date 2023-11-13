# Purpose

Bootstrap knowledge of LLMs ASAP. With a bias/focus to GPT.

Avoid being a link dump. Try to provide only valuable well tuned information.

## Prelude

Neural network links before starting with transformers.

* https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi
* https://www.3blue1brown.com/topics/neural-networks
* http://neuralnetworksanddeeplearning.com/
* https://distill.pub/

## Key

* 🟢 = easy, 🟠 = medium, 🔴 = hard
* 🕰️ = long, 🙉 = low quality audio

## Youtube Lessons

* 🟢🕰️ **Łukasz Kaiser** [Attention is all you need; Attentional Neural Network Models](https://www.youtube.com/watch?v=rBCqOTEfxvg) This talk is from 6 years ago.
* 🟢🕰️ **Andrej Karpathy** [The spelled-out intro to language modeling: building makemore](https://www.youtube.com/watch?v=PaCmpygFfXo): basic. bi-gram name generator model by counting, then by NN. using pytorch.
* 🟢🕰️ **Andrej Karpathy**  [Building makemore Part 2: MLP](https://www.youtube.com/watch?v=TCH_1BHY58I): 
* 🕰️ **Andrej Karpathy**  [Building makemore Part 3: Activations & Gradients, BatchNorm](https://www.youtube.com/watch?v=P6sfmUTpUmc)): 
* 🕰️ **Andrej Karpathy**  [Building makemore Part 4: Becoming a Backprop Ninja](https://www.youtube.com/watch?v=q8SA3rM6ckI): 
* 🟢 **Hedu AI** [Visual Guide to Transformer Neural Networks - (Episode 1) Position Embeddings](https://www.youtube.com/watch?v=dichIcUZfOw): Tokens are embedded into a semantic space. sine/cosine position encoding explained very well.
* 🟢 **Hedu AI** [Visual Guide to Transformer Neural Networks - (Episode 2) Multi-Head & Self-Attention](https://www.youtube.com/watch?v=mMa2PmYJlCo): Clear overview of multi-head attention.
* 🟢 **Hedu AI** [Visual Guide to Transformer Neural Networks - (Episode 3) Decoder’s Masked Attention](https://www.youtube.com/watch?v=gJ9kaJsE78k): Further details on the transformer architecture.
* 🟠🕰️ **Andrej Karpathy**  [Andrej Karpathy - Let's build GPT: from scratch, in code, spelled out.](https://www.youtube.com/watch?v=kCc8FmEb1nY): build up a Shakespeare gpt-2-like from scratch. starts with bi-gram and adds features one by one. pytorch.
* 🔴🕰️ **Chris Olah** [CS25 I Stanford Seminar - Transformer Circuits, Induction Heads, In-Context Learning](https://www.youtube.com/watch?v=pC4zRb_5noQ): Interpretation. Deep look into the mechanics of induction heads. [Companion article](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html)
* 🟢 **Jay Alammar** [The Illustrated Word2vec - A Gentle Intro to Word Embeddings in Machine Learning](https://www.youtube.com/watch?v=ISPId9Lhc1g)
* 🟢 **Jay Alammar** [How GPT3 Works - Easily Explained with Animations](https://www.youtube.com/watch?v=MQnJZuBGmSQ): extremely high level basic overview.
* 🟢🕰️ **Jay Alammar** [The Narrated Transformer Language Model](https://www.youtube.com/watch?v=-QH8fRhqFHM): much deeper look at the architecture. goes into detail. [Companion article](https://jalammar.github.io/illustrated-transformer/).
* **Sebastian Raschka** [L19: Self-attention and transformer networks](https://sebastianraschka.com/blog/2021/dl-course.html#l19-self-attention-and-transformer-networks) Academic style lecture series on self-attention transformers
* 🟢🕰️🙉 **Mark Chen** [Transformers in Language: The development of GPT Models including GPT3](https://www.youtube.com/watch?v=qGkzHFllWDY) A chunk of this lecture is about applying GPT to images. Same lecture series as the Chris Olah one. [Rest of the series](https://www.youtube.com/playlist?list=PLoROMvodv4rNiJRchCzutFw5ItR_Z27CM). Papers listed in the talk:
   * "GPT-1": **Liu et. al.** [Generating Wikipedia by Summarizing Long Sequences](https://arxiv.org/abs/1801.10198)
   * "GPT-2": **Radford et. al.** [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) [github.com/openai/gpt-2](https://github.com/openai/gpt-2) [OpenAI: Better Language Models](https://openai.com/research/better-language-models) [Fermats Library](https://www.fermatslibrary.com/s/language-models-are-unsupervised-multitask-learners)
   * "GPT-3": **Brown et. al.** [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) (I think this is it, can't find the quoted text inside this paper)

# Articles

* 🟠 **Jay Mody** [GPT in 60 Lines of NumPy](https://jaykmody.com/blog/gpt-from-scratch/)
* 🟠 **PyTorch** [Language Modeling with nn.Transformer and TorchText](https://pytorch.org/tutorials/beginner/transformer_tutorial.html)
* 🟠 **Sasha Rush et. al.** [The Annotated Transformer](http://nlp.seas.harvard.edu/annotated-transformer/)
* 🟢 **Jay Alammar** [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) companion video above.
* 🔥 **Chris Olah et. al.** [In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) companion video lecture above

# Research Paper Lists

* **Sebastian Raschka** [Understanding Large Language Models -- A Transformative Reading List](https://sebastianraschka.com/blog/2023/llm-reading-list.html) This article lists some of the most important papers in the area.
* **OpenAI** [Research Index](https://openai.com/research)

# Research Papers

* **Radford et. al.** [Improving Language Understanding by Generative Pre-Training (2018)](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) a page accompanying this paper on the OpenAI blog [Improving language understanding with unsupervised learning](https://openai.com/research/language-unsupervised)
* **Kaplan et. al.** [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) A variety of models were trained using varying amounts of compute, data set size, and number of parameters. This enables us to predict what parameters will work well in larger future models. See also **Gwern Branwen** [The Scaling Hypothesis](https://gwern.net/scaling-hypothesis)

# Philosophy of GPT

* **Isaac Asimov** [The Last Question (1956)](http://users.ece.cmu.edu/~gamvrosi/thelastq.html)
* **Justin Weinberg, Daily Nous** [Philosophers On GPT-3](https://dailynous.com/2020/07/30/philosophers-gpt-3/)
* **Fernando Borretti** [And Yet It Understands](https://borretti.me/article/and-yet-it-understands)
* **Ted Chiang** [ChatGPT Is a Blurry JPEG of the Web](https://www.newyorker.com/tech/annals-of-technology/chatgpt-is-a-blurry-jpeg-of-the-web)
* **Noam Chomsky** [The False Promise of ChatGPT](https://www.nytimes.com/2023/03/08/opinion/noam-chomsky-chatgpt-ai.html)

*This page is not finished yet. I will continue adding to this.*
