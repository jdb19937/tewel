#!/usr/bin/perl

$| = 1;
@ARGV or die "Usage: $0 pic.any\n";
my $fn = shift(@ARGV);

if (grep $fn =~ /\.$_$/, qw(avi mkv mp4)) {
  my $r = @ARGV ? int($ARGV[0]) : 0;

  exec('ffmpeg',
    '-hide_banner', '-loglevel', 'quiet', '-nostats', '-nostdin',
    $r ? ('-r', $r) : ( ),
    '-i', $fn,
    '-an', '-f', 'rawvideo', '-vcodec', 'ppm',
    '-y', '/dev/stdout'
  );
  die "$0: ffmpeg: $!\n";
} else {
  exec('convert', $fn, "-depth", '8', '+set', 'comment', 'ppm:-');
  die "$0: convert: $!\n";
}