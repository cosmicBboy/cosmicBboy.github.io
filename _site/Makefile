JEKYLL_SCRIPT=https://gist.githubusercontent.com/mcwitt/eaec84a6b50ad5ac9fb2/raw/dffb4fdc458d6ce089dc2a1d7372ae4eafa99534/jekyll-post.py
JEKYLL_TEMPLATE=https://gist.githubusercontent.com/mcwitt/eaec84a6b50ad5ac9fb2/raw/dffb4fdc458d6ce089dc2a1d7372ae4eafa99534/jekyll-post.tpl#
BLOG_NOTEBOOKS=notebooks
# current date
DATE ?= `date +%Y-%m-%d`

requirements:
	brew install wget
	mkdir -p ./$(BLOG_NOTEBOOKS)

serve:
	# make sure to use python 2.7
	bundle exec jekyll serve

serve-drafts:
	bundle exec jekyll serve --drafts

setup-ipynb:
	wget -P ./$(BLOG_NOTEBOOKS) $(JEKYLL_SCRIPT)
	wget -P ./$(BLOG_NOTEBOOKS) $(JEKYLL_TEMPLATE)

nbconvert-draft:
	jupyter nbconvert --config jekyll-post.py $(NOTEBOOK) \
		--to=markdown && \
	mv $$(dirname $(NOTEBOOK))/$$(basename $(NOTEBOOK) .ipynb).md _drafts

nbconvert-pub:
	jupyter nbconvert --config jekyll-post.py $(NOTEBOOK) \
		--to=markdown && \
	mv $$(dirname $(NOTEBOOK))/$$(basename $(NOTEBOOK) .ipynb).md _posts
