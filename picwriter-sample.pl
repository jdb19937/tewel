#!/usr/bin/perl

@ARGV or die "Usage: $0 pic.any\n";

exec('convert', 'ppm:-', $ARGV[0]);
die "$0: convert: $!\n";
