"use client";

import * as Select from "@radix-ui/react-select";
import * as Slider from "@radix-ui/react-slider";
import * as Switch from "@radix-ui/react-switch";
import * as ToggleGroup from "@radix-ui/react-toggle-group";
import { Check, ChevronDown } from "lucide-react";
import { useId } from "react";
import { cn } from "@/lib/utils";

export function Field({
  label, hint, htmlFor, children, className,
}: { label: string; hint?: string; htmlFor?: string; children: React.ReactNode; className?: string }) {
  return (
    <div className={cn("flex flex-col gap-1.5", className)}>
      <label htmlFor={htmlFor} className="text-[13px] font-medium text-ink">{label}</label>
      {children}
      {hint && <p className="text-xs text-ink-faint">{hint}</p>}
    </div>
  );
}

const segItem =
  "flex-1 whitespace-nowrap rounded-md px-2.5 py-1.5 text-[13px] text-ink-soft transition-colors " +
  "hover:text-ink data-[state=on]:bg-ink data-[state=on]:text-paper disabled:opacity-40";

export function Segmented<T extends string>({
  label, value, options, onChange, hint,
}: { label: string; value: T; options: { value: T; label: string }[]; onChange: (v: T) => void; hint?: string }) {
  return (
    <Field label={label} hint={hint}>
      <ToggleGroup.Root
        type="single" value={value} aria-label={label}
        onValueChange={(v) => v && onChange(v as T)}
        className="flex gap-1 rounded-lg border border-line bg-field p-1"
      >
        {options.map((o) => (
          <ToggleGroup.Item key={o.value} value={o.value} className={segItem}>{o.label}</ToggleGroup.Item>
        ))}
      </ToggleGroup.Root>
    </Field>
  );
}

export function MultiToggle<T extends string>({
  label, value, options, onChange, hint,
}: { label: string; value: T[]; options: { value: T; label: string }[]; onChange: (v: T[]) => void; hint?: string }) {
  return (
    <Field label={label} hint={hint}>
      <ToggleGroup.Root
        type="multiple" value={value} aria-label={label}
        onValueChange={(v) => v.length && onChange(v as T[])}
        className="flex flex-wrap gap-1.5"
      >
        {options.map((o) => (
          <ToggleGroup.Item
            key={o.value} value={o.value}
            className="rounded-full border border-line bg-field px-3 py-1 text-[13px] text-ink-soft hover:text-ink data-[state=on]:border-ink data-[state=on]:bg-ink data-[state=on]:text-paper"
          >
            {o.label}
          </ToggleGroup.Item>
        ))}
      </ToggleGroup.Root>
    </Field>
  );
}

export function NumberSlider({
  label, value, min, max, step = 1, onChange, format, hint, disabled, log,
}: {
  label: string; value: number; min: number; max: number; step?: number;
  onChange: (v: number) => void; format?: (v: number) => string; hint?: string; disabled?: boolean;
  /** Logarithmic track for long-tailed counts (min must be >= 0). */
  log?: boolean;
}) {
  const id = useId();
  const L = Math.log1p(max);
  const toPos = (v: number) => (log ? (Math.log1p(v) / L) * 1000 : v);
  const fromPos = (p: number) => (log ? Math.max(min, Math.round(Math.expm1((p / 1000) * L))) : p);
  return (
    <div className={cn("flex flex-col gap-1.5", disabled && "opacity-45")}>
      <div className="flex items-baseline justify-between gap-3">
        <label htmlFor={id} className="text-[13px] font-medium text-ink">{label}</label>
        <span className="tabular font-display text-[15px] font-medium text-ink">{format ? format(value) : value}</span>
      </div>
      <Slider.Root
        id={id} value={[toPos(value)]} min={log ? toPos(min) : min} max={log ? 1000 : max}
        step={log ? 1 : step} disabled={disabled}
        onValueChange={([p]) => onChange(fromPos(p))} aria-label={label}
        className="relative flex h-5 w-full touch-none select-none items-center"
      >
        <Slider.Track className="relative h-1 grow rounded-full bg-line">
          <Slider.Range className="absolute h-full rounded-full bg-ink" />
        </Slider.Track>
        <Slider.Thumb aria-label={label} aria-valuetext={format ? format(value) : String(value)} className="block h-4 w-4 rounded-full border-2 border-ink bg-field shadow-sm" />
      </Slider.Root>
      {hint && <p className="text-xs text-ink-faint">{hint}</p>}
    </div>
  );
}

export function SwitchField({
  label, checked, onChange, hint, disabled,
}: { label: string; checked: boolean; onChange: (v: boolean) => void; hint?: string; disabled?: boolean }) {
  const id = useId();
  return (
    <div className={cn("flex items-start justify-between gap-4", disabled && "opacity-45")}>
      <div>
        <label htmlFor={id} className="text-[13px] font-medium text-ink">{label}</label>
        {hint && <p className="text-xs text-ink-faint">{hint}</p>}
      </div>
      <Switch.Root
        id={id} checked={checked} onCheckedChange={onChange} disabled={disabled}
        className="relative h-6 w-10 shrink-0 rounded-full border border-line bg-field data-[state=checked]:border-ink data-[state=checked]:bg-ink"
      >
        <Switch.Thumb className="block h-4 w-4 translate-x-1 rounded-full bg-ink-faint transition-transform data-[state=checked]:translate-x-[19px] data-[state=checked]:bg-paper" />
      </Switch.Root>
    </div>
  );
}

export function SelectField<T extends string>({
  label, value, options, onChange, hint,
}: { label: string; value: T; options: { value: T; label: string }[]; onChange: (v: T) => void; hint?: string }) {
  const id = useId();
  return (
    <Field label={label} hint={hint} htmlFor={id}>
      <Select.Root value={value} onValueChange={(v) => onChange(v as T)}>
        <Select.Trigger
          id={id}
          className="inline-flex h-9 items-center justify-between rounded-lg border border-line bg-field px-3 text-[14px] text-ink"
        >
          <Select.Value />
          <Select.Icon><ChevronDown size={16} aria-hidden className="text-ink-faint" /></Select.Icon>
        </Select.Trigger>
        <Select.Portal>
          <Select.Content
            position="popper" sideOffset={4}
            className="z-50 max-h-72 min-w-[var(--radix-select-trigger-width)] overflow-hidden rounded-lg border border-line bg-field text-ink shadow-lg"
          >
            <Select.Viewport className="p-1">
              {options.map((o) => (
                <Select.Item
                  key={o.value} value={o.value}
                  className="relative flex cursor-default select-none items-center rounded-md py-1.5 pl-7 pr-3 text-[14px] outline-none data-[highlighted]:bg-line"
                >
                  <Select.ItemIndicator className="absolute left-2"><Check size={14} aria-hidden /></Select.ItemIndicator>
                  <Select.ItemText>{o.label}</Select.ItemText>
                </Select.Item>
              ))}
            </Select.Viewport>
          </Select.Content>
        </Select.Portal>
      </Select.Root>
    </Field>
  );
}
