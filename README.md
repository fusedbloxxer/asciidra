# asciidra

Convert Images to ASCII art using deep learning optimization methods

```shell
uv run python -m asciidra                              \
 --font /usr/share/fonts/noto/NotoSansMono-Regular.ttf \
 --chars resources/charsets/complex.txt                \
 --input resources/images/reze.jpg                     \
 --glyphs resources/glyphs                             \
 --output resources/output                             \
 --device 'cuda:1'                                     \
 --max-steps 300                                       \
 --eps 0.005                                           \
 --eta 1.25                                            \
 --tau 0.95                                            \
 --seed 42                                             \
 --pad 0.0                                             \
 --size 8
```
