# asciidra

Convert Images to ASCII art using deep learning optimization methods

```shell
uv run python -m asciidra                              \
 --font /usr/share/fonts/noto/NotoSansMono-Regular.ttf \
 --input resources/images/tea.jpg                      \
 --chars resources/charsets/complex.txt                \
 --glyphs resources/glyphs                             \
 --output resources/output                             \
 --max-steps 1000                                      \
 --tau 0.75                                            \
 --size 8                                              \
 --device 'cuda:0'                                     \
 --pad 0.5
```

```shell
uv run python -m asciidra                              \
 --font /usr/share/fonts/noto/NotoSansMono-Regular.ttf \
 --input resources/images/jahy.jpg                     \
 --chars resources/charsets/simple.txt                 \
 --glyphs resources/glyphs                             \
 --output resources/output                             \
 --max-steps 1000                                      \
 --tau 0.95                                            \
 --size 8                                              \
 --pad 0.5
```

```shell
uv run python -m asciidra                              \
 --font /usr/share/fonts/noto/NotoSansMono-Regular.ttf \
 --input resources/images/jahy.jpg                     \
 --chars resources/charsets/simple.txt                 \
 --glyphs resources/glyphs                             \
 --output resources/output                             \
 --max-steps 1000                                      \
 --tau 0.95                                            \
 --size 8                                              \
 --pad 0.5
```
