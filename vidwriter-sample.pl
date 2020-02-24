#!/usr/bin/perl -s

@ARGV or die "Usage: $0 [-r=rate] vid.any\n";

$::r ||= 16;

exec('ffmpeg',
  '-hide_banner', '-loglevel', 'quiet', '-nostats', '-nostdin',
  '-r', $::r,
  '-i', '/dev/stdin',
  '-r', $::r,
  '-y', $ARGV[0]
);
die "$0: ffmpeg: $!\n";
