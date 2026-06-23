# mingfeima.github.io

Personal GitHub Pages site for [mingfeima](https://github.com/mingfeima).

Live URL (after deployment): https://mingfeima.github.io

## Posts

Source: [`posts/cpu-gdn-chunk-kernel.md`](posts/cpu-gdn-chunk-kernel.md)  
Published: [Optimizing GDN Chunk Prefill on Intel AMX CPU](https://mingfeima.github.io/posts/cpu-gdn-chunk-kernel.html)

Posts are Markdown + [Jekyll](https://jekyllrb.com/) (GitHub Pages builds them automatically).

## Local preview

```bash
cd ~/mingfeima.github.io
bundle install   # first time only
bundle exec jekyll serve
```

Open http://localhost:4000

Static preview without Jekyll (home page only):

```bash
python3 -m http.server 8080
```

## Deploy

Push to the `main` branch of the `mingfeima/mingfeima.github.io` repository.
GitHub Pages will publish automatically.
