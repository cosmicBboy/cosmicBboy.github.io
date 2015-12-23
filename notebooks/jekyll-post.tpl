{% extends 'markdown.tpl' %}

{%- block header -%}
---
layout: post
title: "{{resources['metadata']['name']}}"
tags:
    - python
    - notebook
---
{%- endblock header -%}

{% block input %}
{{ '{% highlight python %}' }}
{{ cell.source }}
{{ '{% endhighlight %}' }}
{% endblock input %}

{% block data_png %}
![png]({{ output.metadata.filenames['image/png'] | path2url }})
{% endblock data_png %}

{% block data_jpg %}
![jpeg]({{ output.metadata.filenames['image/jpeg'] | path2url }})
{% endblock data_jpg %}
