#fname=Notes_about_network_propagation
#baname=BachelorThesis
#baname=thesis
#Notes_about_network_propagation.pdf: Notes_about_network_propagation.tex mybib.bib
#	pdflatex $(fname)
#	biber $(fname)
#	pdflatex $(fname)
name = topic_list
slides = slides_test
poster = poster_test
debug = debug


topic_list.pdf:
	#pdflatex $(name)
	#biber $(name)
	#pdflatex $(name)
	#pdflatex -interaction=nonstopmode -halt-on-error -shell-escape $(name)
	xelatex -interaction=nonstopmode -halt-on-error -shell-escape $(name)
	biber $(name)
	#pdflatex -interaction=nonstopmode $(name)
	xelatex -interaction=nonstopmode $(name)

slides_test.pdf:
	xelatex -interaction=nonstopmode -halt-on-error -shell-escape $(slides)
	#pdflatex -interaction=nonstopmode -halt-on-error -shell-escape $(slides)
	#pdflatex $(slides)
	biber $(slides)
	#pdflatex $(slides)
	xelatex $(slides)
	

poster_test.pdf:
	#pdflatex -interaction=nonstopmode -halt-on-error -shell-escape $(poster)
	xelatex -interaction=nonstopmode -halt-on-error -shell-escape $(poster)
	#pdflatex $(poster)
	biber $(poster)
	#pdflatex $(poster)
	xelatex $(poster)

debug.pdf:
	xelatex -interaction=nonstopmode -halt-on-error -shell-escape $(debug)
	#pdflatex -interaction=nonstopmode -halt-on-error -shell-escape $(debug)
	biber $(debug)
	xelatex $(debug)
	#pdflatex $(debug)
