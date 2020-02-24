#!/usr/bin/perl

exec('convert', 'ppm:-', $ARGV[0]);
die "$0: convert: $!\n";
