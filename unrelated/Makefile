cv=cv2022
letter=motivation

cv:
	lualatex -interaction=nonstopmode -halt-on-error -shell-escape $(cv)
	biber $(cv)
	lualatex -interaction=nonstopmode $(cv)

motivation:
	lualatex -interaction=nonstopmode -halt-on-error -shell-escape $(letter)
	biber $(letter)
	lualatex -interaction=nonstopmode $(letter)
