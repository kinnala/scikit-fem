all: ex01_solution.png \
     ex02_solution.png \
     ex05_solution.png \
     ex06_solution.png \
     ex17_solution.png \
     ex20_stream-lines.png \
     ex20_velocity-vectors.png \
     ex23.png \
     ex25.png \
     ex26_solution.png \
     ex27-150.0-psi.png \
     ex27-450.0-psi.png \
     ex27-750.0-psi.png
%.png: %.py
	python $<
%_solution.png: %.py
	python $<
ex20_stream-lines.png ex20_velocity-vectors.png: ex20.py
	python $<
ex27-150.0-psi.png ex27-450.0-psi.png ex27-750.0-psi.png: ex27.py
	python $<
clean:
	${RM} *.png *.vtk *~
.PHONY: all clean
