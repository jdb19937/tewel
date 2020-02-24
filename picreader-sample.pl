#!/usr/bin/perl

exec('convert', $ARGV[0], "-depth", '8', '+set', 'comment', 'ppm:-');
die "$0: convert: $!\n";
