CXX = mpiCC
CXXFLAGS = -std=c++17
INCLUDES = -I/scratch/jh2/hs9158/gizmo-fork/gizmo2/slug2/src/ \
           -I/scratch/jh2/hs9158/gizmo-fork/gizmo2/slug2/src/pdfs \
           -I/scratch/jh2/hs9158/gizmo-fork/gizmo2/src/galaxy_sf/
LIBDIRS = -L/scratch/jh2/hs9158/gizmo-fork/gizmo2/slug2/src/
LIBS = -lhdf5 -lhdf5_cpp -lgsl -lgslcblas -lslug

SRCS = SlugAnalysis.cpp /scratch/jh2/hs9158/gizmo-fork/gizmo2/src/galaxy_sf/slug_wrapper.cpp
TARGET = SlugAnalysis

$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(LIBDIRS) -o $@ $^ $(LIBS)

clean:
	rm -f $(TARGET)
