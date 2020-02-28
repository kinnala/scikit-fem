all: ex01_solution.png \
     ex02_solution.png \
     ex04_solution.png \
     ex05_solution.png \
     ex06_solution.png \
     ex17_solution.png \
     ex19.gif \
     ex20_stream-lines.png ex20_velocity-vectors.png \
     ex23.png \
     ex25.png \
     ex26_solution.png \
     ex27-150.0-psi.png ex27-450.0-psi.png ex27-750.0-psi.png \
     ex28.png ex28-inlet-outlet.png \
     ex29.png \
     ex30_stream-lines.png \
     ex31_eigenmode.png
%.png: %.py
	python $<
%_solution.png: %.py
	python $<
%_eigenmode.png: %.py
	python $<
%.gif: %.py
	python $< --gif
%_stream-lines.png: %.py
	python $<
ex20_velocity-vectors.png: ex20_stream-lines.png
ex27-750.0-psi.png: ex27.py
	python $<
ex27-150.0-psi.png ex27-450.0-psi.png: ex27-750.0-psi.png
ex28-inlet-outlet.png: ex28.png
clean:
	${RM} *.png *.vtk *.gif *~
.PHONY: all clean
