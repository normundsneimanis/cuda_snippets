/*
 * Salt-and-pepper noise filtering on GPU
 * Arnis Lektauers 2019 Riga Technical University
 */
#pragma once

#define FILTER_RADIUS	3
#define EDGE_SIZE		FILTER_RADIUS / 2
#define WINDOW_SIZE		FILTER_RADIUS * FILTER_RADIUS