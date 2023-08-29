FROM python:3.9-slim

RUN apt-get update \
  && apt-get install -y gcc git wget \
  && apt-get clean

RUN pip install --upgrade pip
RUN pip install --no-cache-dir poetry

RUN sh -c "$(wget -O- https://github.com/deluan/zsh-in-docker/releases/download/v1.1.5/zsh-in-docker.sh)" -- \
  -p git \
  -p https://github.com/zsh-users/zsh-autosuggestions \
  -p https://github.com/zsh-users/zsh-completions \
  -p https://github.com/zsh-users/zsh-history-substring-search \
  -p https://github.com/zsh-users/zsh-syntax-highlighting \
  -p 'history-substring-search' \
  -a 'bindkey "\$terminfo[kcuu1]" history-substring-search-up' \
  -a 'bindkey "\$terminfo[kcud1]" history-substring-search-down'

WORKDIR /workspaces/wine_quality

ENV SHELL=/bin/zsh
ENTRYPOINT [ "/bin/zsh" ]
