#!/usr/bin/perl

@ARGV or die "Usage: $0 [-r=rate] vid.any\n";

exec('ffmpeg',
  '-hide_banner', '-loglevel', 'quiet', '-nostats', '-nostdin',
  $::r ? ('-r', $::r) : ( ),
  '-i', $ARGV[0],
  '-an', '-f', 'rawvideo', '-vcodec', 'ppm',
  $::r ? ('-r', $::r) : ( ),
  '-y', '/dev/stdout'
);
die "$0: ffmpeg: $!\n";
