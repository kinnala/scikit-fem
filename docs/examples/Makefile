all: ex01_solution.png \
     ex02_solution.png \
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
     ex29.png
%.png: %.py
	python $<
%_solution.png: %.py
	python $<
%.gif: %.py
	python $< --gif
ex20_velocity-vectors.png: ex20.py
	python $<
ex20_stream-lines.png: ex20_velocity-vectors.png
ex27-750.0-psi.png: ex27.py
	python $<
ex27-150.0-psi.png ex27-450.0-psi.png: ex27-750.0-psi.png
clean:
	${RM} *.png *.vtk *.gif *~
.PHONY: all clean
