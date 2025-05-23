# CUDA 12.8.
#FROM nvcr.io/nvidia/pytorch:25.04-py3

#RUN pip install evaluate transformers[torch] datasets sentence-transformers
# Instale dependências necessárias, incluindo g++ e gcc
#RUN apt-get update && apt-get install -y \
#    build-essential \
#    python3-pip \
#    git \
#    && rm -rf /var/lib/apt/lists/*

# Atualiza pip e instala PyTorch compatível com CUDA 11.8
#RUN pip install --upgrade pip && \
#    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

#RUN pip install nltk tqdm pyarrow wandb matplotlib pytorch_metric_learning schedulefree
#RUN pip install transformers sentence-transformers tokenizers pandas datasets seaborn sentencepiece
# Instalação do flash-attn
#RUN pip install ninja
#RUN pip install flash-attn --no-build-isolation
#
#RUN pip install faiss-gpu accelerate==1.3.0
#RUN apt update && apt install tmux git -y

#RUN echo "source /etc/profile.d/bash_completion.sh" >> ~/.bashrc

# Crie um diretório de trabalho
#WORKDIR /home/

# Exemplo: Copie seus arquivos e scripts para o container
# COPY . /home/

# Comando padrão para executar o container
#CMD ["/bin/bash", "-l"]
#########################
#FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime
FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel
RUN apt-get update && \
      apt-get -y install sudo build-essential
RUN apt install bash-completion
RUN pip install ninja evaluate seqeval transformers[torch] datasets sentence-transformers protobuf sentencepiece
RUN pip install flash-attn --no-build-isolation
#RUN useradd -rm -d /home/ubuntu -s /bin/bash -g root -G sudo -u 1010 ubuntu
#USER root
#RUN chown -R ubuntu:root /home/ubuntu
#USER ubuntu
USER root
WORKDIR /home/ubuntu
#WORKDIR /home/