---
layout: post
title: "A Human's Guide to Algorithmic Bias"
tags:
    - python
    - notebook
---
For the past year I've been doing a lot of introspecting, reading, and
critical thinking about bias in algorithms, particularly of the machine-learned variety.
And no, I don't mean bias in terms of the [bias-variance decomposition](https://en.wikipedia.org/wiki/Bias%E2%80%93variance_tradeoff){:target="_blank"}.
I mean bias in the sense that *we're all gonna be screwed if we don't
start thinking about machine learning as a tool for socioeconomic
and political control*.

This control can be explicit, for example in the infamous
case of the [COMPAS recidivism risk model](https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing){:target="_blank"}, which tells decision-makers
the inmates who are likely to re-offend. Or it can be subtle, like when
Facebook determines what kinds of posts to show in your feed, or when Amazon
recommends products for you to buy.

Applications on this spectrum of control or influence, depending on how you
want to look at it, have a dark side that will bear itself out as we continue
to develop machine-learning-driven tools in the context of medicine, social
media, labor, education, and politics.

I've written about this kind of machine learning bias in the context of
[responsible data science](http://techblog.arena.io/Responsible_Data_Science/){:target="_blank"},
but in this post I'll articulate a general framework that I hope will help you navigate
the world of algorithmic applications.

Before getting into that, let's talk about the problem.

# Teaching Silicon Children

<img src="https://data.whicdn.com/images/42842083/large.png" width=750>
> Haley Joel Osment in the 2001 film "A.I. Artificial Intelligence"

Most machine learning today comes in the form of what's called *supervised learning*.
You can think of supervised learning as us, the human parents, teaching our computationally
gifted, silicon-based children how to recognize, process, and label patterns within a
limited range of real-world phenomena. In other words, this class of algorithm learns
a mapping from some input to some desired output.

**Consider the following tasks:**

1. "Read a person's facebook feed (`input`) and tell me whether she's likely
   to be a registered Republican or Democrat (`output`)".
2. "Given neighborhood-level drug-related arrests (`input`), tell me
   the neighorhoods with the highest chance of drug-related crime (`output`)".
3. "Look at this image of a face (`input`) and tell me if she's gay or not (`output`)"
   (btw this is a [real thing](https://osf.io/zn79k/){:target="_blank"}).

The parent-child analogy captures the fact that these machine learning systems are
still narrow in scope and cognitive ability. They might out-perform the best of us at
[diagnosing pneumonia](https://news.stanford.edu/2017/11/15/algorithm-outperforms-radiologists-diagnosing-pneumonia/){:target="_blank"},
but any single machine learning system currently doesn't yet possess the faculties to
reason about a wider range of concerns such as thinking about the ethical implications
of its own predictions.

# Modern Physiognomy

<img src="https://upload.wikimedia.org/wikipedia/commons/a/a6/Physiognomy.jpg" width="750"/>

> **Physiognomy**, characterized by some as a pseudoscience, is "the assessment of character or
personality from a person's outer appearance, especially the face". *Source: [Wikipedia](https://en.wikipedia.org/wiki/Physiognomy)*.

In this sense, we can think of machine learning in the social context as a modern, more
sophisticated version of physiognomy. This is because applying machine
learning to socioeconomic and political systems is essentially the exercise of gathering
measurable characteristics about people (the `inputs`) and ascribing higher-level classification
systems (the `outputs`) to them based on correlations between the `inputs` and `outputs`.

[Kate Crawford](https://www.microsoft.com/en-us/research/people/kate/){:target="_blank"} talks about this
and bias more broadly in her [keynote talk](https://www.youtube.com/watch?v=fMym_BKWQzk){:target="_blank"}
at this year's NIPS conference, but to put it briefly, the big problem arises when we
start thinking more critically about classification systems and how they're used to
promulgate and solidify a particular social structure.

One notable case is that of [Cesare Lombroso](https://en.wikipedia.org/wiki/Cesare_Lombroso){:target="_blank"},
a 19th century criminologist and physician. He proposed a set of physical features that
characterize a "born criminal", among them, things like large jaws, a low sloping forehead,
and handle-shaped ears.

Although claims like these are problematic in themselves, the material consequences of
classification systems, machine-learned or not, have to do with the *actions*,
*decisions*, or *policies* that follow. Once we're able to formalize and implement
these classification systems in the form of machine learning algorithms, we can effectively
scale out and automate the structural biases that exist in today's society.

# The "Dark Side" and the "Light Side"

<img src="https://pm1.narvii.com/6044/291a529cc63c7b838410a407975d709051e19305_hq.jpg" width=750>
> The primeval struggle between good and evil, as depicted by Darth Vader and Obi Wan Kenobi

To highlight the fact that we can use the very same machine learning system for
benefit or harm, let's recast the three applications that I described above in the
framework of real-world use cases.

1. Read a person's facebook feed and tell me whether she's likely to be
   a registered Republican or Democrat so that I can:
   1. `action`: spread misinformation about the opposing party.
   1. `action`: send them information about the relevant candidate's platform.
1. Given instances of neighborhood-level arrests, tell me
   the neighorhoods with the highest chance of future crime so that
   I can:
   1. `action`: implement a stop-and-frisk policy and increase police presence in those areas.
   1. `action`: increase funding and support for recidivism-prevention programs.
1. Look at this image of a face and tell me if he's gay or not so that I can:
   1. `action`: identify likely candidates to send to "conversion therapy".
   1. `action`: send him relevant information about sexual health.

Notice that the differences in `actions` in each application can be subtle,
and our value judgement about "rightness" or "wrongness" depend on our own
belief systems and how exactly *actions*, *decisions*, and *policies* are
implemented.

Quite honestly, the "dark side" and "light side" is itself a simple binary
classification system that is often too simplistic to apply to real-world
situations. So, as any experienced ethicist can tell you, I think the best
we can do is to continue to ask the right questions and try to act accordingly.

# Question the System

The next time you're using Facebook, Amazon, Netflix, or any other digital product, and
it serves you some sort of prediction about what content you might enjoy, what thing
you might want to buy, or who you might want to connect with, ask yourself:

1. **What do the architects of this system have to gain in making this prediction?**
1. **What historical training data did the app use to make this prediction?**
1. **Are these data correlated with socially sensitive attributes like race?**
1. **Are any socially sensitive attributes correlated with the labels?**
1. **Who labelled the training data, and who decided what the labels are to begin with?**
1. **What kinds of characteristics about me does the system know about?**
1. **Is the system making this prediction because I belong to a particular social group?**
1. **Is the classification system underlying the predicted labels realistic?**
1. **How are my actions on this app being fed back into the machine learning algorithm?**
1. **How are the predictions influencing my own view of the world?**

Think of these as a sort of mental vaccine to immunize yourself from taking for granted
the fact that these useful and convenient services aren't just here to make your life
better. It's all too easy to forget that these apps reflect the hopes, desires, and biases of
its architects, who happen to be human beings too, at least for now...
