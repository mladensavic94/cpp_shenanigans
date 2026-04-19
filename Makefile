CXX      := g++
CXXFLAGS := -std=c++20 -g
LDFLAGS  := -Wl,--build-id

TARGETS  := tokenizer micrograd

all: $(TARGETS)

tokenizer: tokenizer.cpp
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $<

micrograd: micrograd.cpp
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $<

clean:
	rm -f $(TARGETS)

.PHONY: all clean
