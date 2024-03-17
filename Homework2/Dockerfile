FROM ubuntu

WORKDIR /Homework

RUN apt update
RUN apt install -y texlive python3

COPY requirements.txt /Homework/rquirements.txt
COPY generate.py /Homework/generate.py
COPY main.py /Homework/main.py
COPY progression.png /Homework/progression.png
COPY graph.png /Homework/graph.png

RUN apt install -yq python3.10-venv
RUN apt install -yq python3.10
RUN python3 -m venv .venv

RUN .venv/bin/python -m pip install --no-cache-dir -r rquirements.txt
RUN .venv/bin/python -m pip install --index-url https://test.pypi.org/simple/ --no-deps TexString

RUN mkdir artifacts
RUN .venv/bin/python generate.py

CMD ["pdflatex", "-output-directory=data", "artifacts/table.tex"]
