JEKYLL_SCRIPT=https://gist.githubusercontent.com/mcwitt/eaec84a6b50ad5ac9fb2/raw/dffb4fdc458d6ce089dc2a1d7372ae4eafa99534/jekyll-post.py
JEKYLL_TEMPLATE=https://gist.githubusercontent.com/mcwitt/eaec84a6b50ad5ac9fb2/raw/dffb4fdc458d6ce089dc2a1d7372ae4eafa99534/jekyll-post.tpl#
BLOG_NOTEBOOKS=notebooks

requirements:
	brew install wget
	mkdir -p ./$(BLOG_NOTEBOOKS)

setup-ipynb:
	wget -P ./$(BLOG_NOTEBOOKS) $(JEKYLL_SCRIPT)
	wget -P ./$(BLOG_NOTEBOOKS) $(JEKYLL_TEMPLATE)

nbconvert:
	cd ./$(BLOG_NOTEBOOKS) && bash nbconvert.sh
