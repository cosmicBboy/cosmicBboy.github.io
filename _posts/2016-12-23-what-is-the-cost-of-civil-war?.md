---
layout: post
title: "What is the Cost of Civil War?"
tags:
    - python
    - notebook
---
I recently became motivated to engage more with the civil war in Syria when I learned that [the last hospital in Eastern Aleppo is no more](http://www.npr.org/sections/goatsandsoda/2016/11/28/503022907/the-last-hospital-in-eastern-aleppo-is-no-more){:target="_blank"}. The question of how to engage with issues that induce outrage, anxiety and a slew of other emotions is itself an interesting one, and I imagine that different people have different ways of responding to these kinds of news stories. However, this post won't be about this question, but another:

#### What's the total economic damage caused by the Syrian civil war?

![Image](https://raw.githubusercontent.com/cosmicBboy/syria-project/master/eda/plots/aleppo_2016.png)

### 12,000 of the 160,000 buildings in Aleppo have been damaged

To make this comparison, I used [neighborhood damage data](https://data.humdata.org/dataset/geodata-of-neighborhood-damage-percentages-aleppo-syria-february-18-2016) that compares satellite imagery of pre-conflict snapshots from 2009 to post-conflict snapshots from 2010 - 2015. As you can see, much of the damage was done in highly populated neighborhoods. Considering the resources and effort needed to conduct a war, this pattern tells me that the collective tactical decisions made by the parties involved have resulted in some local maximum of harm exacted upon the largest possible group of people.

### $2.3 billion: total estimated cost of damages done to the economy

Based on the estimates in the [global exposure dataset](https://data.humdata.org/dataset/gar15-global-exposure-dataset-for-syrian-arab-republic), the total value of capital stock in the city of Aleppo was \$29.5 billion in 2015. Assuming that the percentage of damaged buildings (8%) in each neighborhood is a fairly reasonable proxy for economic disruption and infrastructure damage, we can project a lower bound of $2.3 billion as the total cost of the Syrian civil war to Aleppo's economy. This seems like a staggering amount, not even considering the immeasurable human suffering endured by the its citizens.

### Parting thoughts

An interesting future inquiry is to compare this amount to other estimates about the cost of destructive events, like natural disasters or other conflicts. Much statistical work needs to be done on these datasets to measure how confident we are in their reliability, but for now I'll leave you with these thoughts.

From a data science perspective, questions like the one I posed in the beginning are hard to answer and depend heavily on what's available in the datasphere. Invariably, the data scientist must tweak the question's scope and be comfortable using proxies and rough estimates to get a sense of what an answer might be.

Exploring the data leads to more questions, and assumptions that we make about the data enable us to come up with more or less satisfactory answers to these questions. In my opinion, this is one of the hallmarks of data science: the endless iteration between question-asking, exploration, and insight using statistical techniques.

Questions? Thoughts? Please post them below!
