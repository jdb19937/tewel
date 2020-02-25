#!/usr/bin/perl

@ARGV or die "Usage: $0 pic.any\n";
my $fn = shift(@ARGV);

if (grep $fn =~ /\.$_$/, qw(avi mkv mp4)) {
  my $r = @ARGV ? $ARGV[0] : 16;

  exec('ffmpeg',
    '-hide_banner', '-loglevel', 'quiet', '-nostats', '-nostdin',
    '-r', $r,
    '-i', '/dev/stdin',
    '-r', $r,
    '-y', $fn
  );
  die "$0: ffmpeg: $!\n";
} else {
  my ($ext) = ($fn =~ /\.(.+$)/);
  $ext ||= 'ppm';

  my $ret = system('convert', 'ppm:-', "$ext:$fn.tmp");
  $ret and die "$0: convert: $!";
  
  $ret = rename("$fn.tmp", $fn);
  $ret or die "$0: rename: $!";

  exit 0;
}
