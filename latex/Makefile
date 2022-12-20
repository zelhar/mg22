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


topic_list.pdf:
	#pdflatex $(name)
	#biber $(name)
	#pdflatex $(name)
	pdflatex -interaction=nonstopmode -halt-on-error -shell-escape $(name)
	biber $(name)
	pdflatex -interaction=nonstopmode $(name)

slides_test.pdf:
	pdflatex -interaction=nonstopmode -halt-on-error -shell-escape $(slides)
	#pdflatex $(slides)
	biber $(slides)
	pdflatex $(slides)
	

poster_test.pdf:
	pdflatex -interaction=nonstopmode -halt-on-error -shell-escape $(poster)
	#pdflatex $(poster)
	biber $(poster)
	pdflatex $(poster)
