#fname=Notes_about_network_propagation
#baname=BachelorThesis
#baname=thesis
#Notes_about_network_propagation.pdf: Notes_about_network_propagation.tex mybib.bib
#	pdflatex $(fname)
#	biber $(fname)
#	pdflatex $(fname)
name = topic_list
slides = slides_test


topic_list.pdf:
	#pdflatex $(name)
	#biber $(name)
	#pdflatex $(name)
	pdflatex -interaction=nonstopmode $(name)
	biber $(name)
	pdflatex -interaction=nonstopmode $(name)

slides_test.pdf:
	pdflatex $(slides)
	biber $(slides)
	pdflatex $(slides)
	
