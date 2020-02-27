#!/usr/bin/perl

$| = 1;
@ARGV or die "Usage: $0 vid.any\n";
my $fn = shift(@ARGV);

if (my $filter = $ENV{TEWEL_VIDREADER_FILTER}) {
  $filter =~ s#^//#/opt/makemore/share/tewel/#;
  open(STDOUT, "| $filter") or die "$0: $filter: $!";
}

my $rate = $ENV{TEWEL_VIDREADER_RATE};

exec('ffmpeg',
  '-hide_banner', '-loglevel', 'quiet', '-nostats', '-nostdin',
  $rate ? ('-r', $rate) : ( ),
  '-i', $fn,
  '-an', '-f', 'rawvideo', '-vcodec', 'ppm',
  '-y', '/dev/stdout'
);
die "$0: ffmpeg: $!\n";
