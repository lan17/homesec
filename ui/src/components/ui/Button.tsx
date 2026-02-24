import type { ButtonHTMLAttributes } from 'react'

type ButtonVariant = 'primary' | 'ghost'

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant
}

export function Button({
  variant = 'primary',
  className,
  type = 'button',
  ...props
}: ButtonProps) {
  const classes = className
    ? `button button--${variant} ${className}`
    : `button button--${variant}`

  return <button type={type} className={classes} {...props} />
}
