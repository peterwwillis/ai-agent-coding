# Makefile

# Simple Makefile for building the Personaplex Vulkan project

# Compiler and flags
CXX = g++
CXXFLAGS = -Wall -Wextra -g

# Build targets
TARGET = personaplex
SOURCES = main.cpp audio_io.cpp controller.cpp

all: $(TARGET)

$(TARGET): $(SOURCES)
	$(CXX) $(CXXFLAGS) -o $@ $^

clean:
	rm -f $(TARGET)