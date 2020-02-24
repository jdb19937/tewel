#!/usr/bin/perl

@ARGV or die "Usage: $0 pic.any\n";

exec('convert', $ARGV[0], "-depth", '8', '+set', 'comment', 'ppm:-');
die "$0: convert: $!\n";
